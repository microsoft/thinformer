# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional

import torch
from torch import nn

# from ...activations import ACT2FN
from transformers.cache_utils import Cache

# from ...generation import GenerationMixin
# from ...integrations import use_kernel_forward_from_hub
# from ...masking_utils import create_causal_mask
# from ...modeling_layers import (
#     GenericForQuestionAnswering,
#     GenericForSequenceClassification,
#     GenericForTokenClassification,
#     GradientCheckpointingLayer,
# )
# from transformers.modeling_outputs import (
#     BaseModelOutputWithPast,
#     CausalLMOutputWithPast,
# )
# from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.utils.deprecation import deprecate_kwarg

# from transformers.utils.generic import check_model_inputs
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from torch.nn.attention.bias import causal_lower_right

from thinformer.triton import ThinformerAttention
from thinformer.timing_utils import multi_events

logger = logging.get_logger(__name__)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # Thinformer Express parameters
        h = kwargs.get("h")
        if h is not None:
            logger.debug(
                f"Using ThinformerAttention with h={h} in LlamaAttention layer {layer_idx}"
            )
        self.h = h
        # Enable PyramidKV's causal_lower_right attention mask.
        self.use_pyramidkv = kwargs.get("use_pyramidkv", False)
        # Express attention sink / sliding-window sizes.
        self.n_first = kwargs.get("n_first", 0)
        self.n_last = kwargs.get("n_last", 0)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if query_states.shape[2] == key_states.shape[2]:
            if self.h is not None:
                # Initialize the ThinformerAttention module
                attn = ThinformerAttention(
                    h=self.h,
                    alias=f"{self.layer_idx}_express",
                    scale=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                    n_first=self.n_first,
                    n_last=self.n_last,
                )

                def attention_interface(
                    module: nn.Module,
                    q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    attention_mask: torch.Tensor,
                    dropout: float,
                    scaling: float,
                    **kwargs: dict,
                ) -> tuple[torch.Tensor, None]:
                    """Attention interface for prefill using ExpressAttention.

                    Args:
                        module: The attention module instance.
                        q: Query tensor of shape (B, H, S, D).
                        k: Key tensor of shape (B, Hkv, S, D).
                        v: Value tensor of shape (B, Hkv, S, D).
                        attention_mask: Causal attention mask (unused, ExpressAttention handles masking internally).
                        dropout: Dropout probability (unused during ExpressAttention).
                        scaling: Scaling factor for attention scores (passed via ExpressAttention constructor).
                        **kwargs: Additional keyword arguments.

                    Returns:
                        Tuple of (attn_output, None) where attn_output has shape (B, S, H, D).

                    """
                    return attn(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        causal=True,
                    )
            else:

                def attention_interface(
                    module: nn.Module,
                    q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    attention_mask: torch.Tensor,
                    dropout: float,
                    scaling: float,
                    **kwargs: dict,
                ) -> tuple[torch.Tensor, None]:
                    """Attention interface for prefill using standard scaled dot-product attention.

                    Args:
                        module: The attention module instance.
                        q: Query tensor of shape (B, H, S, D).
                        k: Key tensor of shape (B, Hkv, S, D).
                        v: Value tensor of shape (B, Hkv, S, D).
                        attention_mask: Causal attention mask (unused, is_causal=True is set instead).
                        dropout: Dropout probability.
                        scaling: Scaling factor for attention scores.
                        **kwargs: Additional keyword arguments.

                    Returns:
                        Tuple of (attn_output, None) where attn_output has shape (B, S, H, D).

                    """
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query=q,
                        key=k,
                        value=v,
                        dropout_p=dropout,
                        is_causal=True,
                        scale=scaling,
                        enable_gqa=True,
                    )
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    return attn_output, None
        else:
            B, _, L, _ = query_states.shape
            _, _, S, _ = key_states.shape
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]
            if hasattr(past_key_values.layers[self.layer_idx], "_log_key_weights"):  # express and balancekv specific
                log_key_weights = torch.zeros(
                    B, query_states.shape[1], L, S,
                    device=query_states.device, dtype=query_states.dtype
                )
                log_key_weights[
                    :,
                    :,
                    :,
                    : past_key_values.layers[self.layer_idx]._log_key_weights.shape[-1],
                ] = repeat_kv(
                    past_key_values.layers[self.layer_idx]._log_key_weights.unsqueeze(
                        2
                    ),
                    self.num_key_value_groups,
                )
                if attention_mask is None:
                    attention_mask = log_key_weights
                else:
                    min_dtype = torch.finfo(query_states.dtype).min
                    attention_mask = torch.where(attention_mask, log_key_weights, min_dtype)
            elif self.use_pyramidkv:
                # HACK: The original attention mask uses the cache size from the the first layer,
                # which may not be correct for subsequent layers if they have different cache sizes.
                # SDPA assumes upper left bias, which is not correct.
                attention_mask = causal_lower_right(L,S)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with multi_events(f"{self.layer_idx}_attn"):
            start.record()
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            end.record()
        attn_output = attn_output.to(query_states.dtype)
        if "query_events" in kwargs and kwargs["query_events"] is not None:
            kwargs["query_events"][str(self.layer_idx)].append((start, end))

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
