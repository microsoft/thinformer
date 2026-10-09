# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers.cache_utils import QuantizedCache

from kvpress.presses.utils import extract_keys_and_values
from kvpress.presses.decoding_press import DecodingPress

from thinformer.triton.express import Express as ExpressModule
from thinformer.timing_utils import multi_events

logger = logging.getLogger(__name__)


@dataclass
class ExpressNew(DecodingPress):
    """
    A press that implements the Express algorithm using the v2 streaming
    module (thinformer.triton.express, algo:express three-sibling form with
    nout / max_halving_rounds parametrization). The `express` press keeps the
    v1 module (thinformer.triton.express_2) for comparison.

    Parameters
    ----------
    base_press : ScorerPress
        The scorer press used to compute importance scores for tokens.
    compression_interval : int, default=10
        Number of decoding steps between compression, i.e. compression will be applied every compression_interval steps.
    target_size : int, default=1024
        Target number of tokens to keep after compression.
        NOTE: for Express, target_size overloads g_eps (log-scale); it is
        converted internally via nout = round(4**target_size).
    hidden_states_buffer_size : int, default=128
        Maximum number of hidden states to keep before compression. Larger values use more GPU memory.
        Note: Some presses don't need buffered hidden states and can set this to 0 to use only the
        current hidden state for compression scoring.
    use_balancekv : bool, default=False
        If True, Express halves with BalanceKV's balanced walk instead of Kernel Halving.
    """

    # Express parameters
    express_states: dict = field(default_factory=dict)
    cache_sizes: dict = field(default_factory=dict)
    g_min: int = 0  # minimum compression level (caps the Compress2 halving depth)
    n_sink: int = 32  # number of initial tokens to always preserve (sink tokens)
    n_window: int = 0  # number of tokens to keep in the window
    use_balancekv: bool = False  # halve with BalanceKV's balanced walk instead of Kernel Halving

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Delegate compression to the base press during decoding phase.

        Args:
            module: The transformer module being compressed
            hidden_states: Buffered hidden states from recent decoding steps (shape: [batch, buffer_len, hidden_dim])
            keys: Key cache from all previous steps including current (shape: [batch, n_heads, seq_len, head_dim])
            values: Value cache from all previous steps including current (shape: [batch, n_heads, seq_len, head_dim])
            attentions: Attention weights (shape varies by implementation)
            kwargs: Additional keyword arguments

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Compressed (keys, values) tensors
        """
        B, H, S, _ = keys.shape
        if S - self.n_sink - self.n_window <= 0:
            compressed_keys = keys
            compressed_values = values
            log_key_weights = None
        else:
            with multi_events(f"{module.layer_idx}_express_prologue"):
                # ignore the stale keys after the sink tokens and the window tokens
                start = self.n_sink + self.express_states[module.layer_idx].size
                end = S - self.n_window
                assert end > start, "end must be greater than start"
                # get the fresh keys and values
                new_keys = keys[:, :, start : end]
                new_values = values[:, :, start : end]

            # compress the recent keys and values
            with multi_events(f"{module.layer_idx}_express_compress"):
                compressed_keys, compressed_values, batch_size, _log_key_weights = self.express_states[module.layer_idx].invoke(
                    new_keys.transpose(1, 2),
                    new_values.transpose(1, 2),
                    alias=str(module.layer_idx),
                    # verbose=(module.layer_idx == 0),
                )

            with multi_events(f"{module.layer_idx}_express_epilogue"):
                compressed_keys = compressed_keys.transpose(1, 2)
                compressed_values = compressed_values.transpose(1, 2)
                if _log_key_weights is None:
                    # The v2 Express returns None weights while it is still in
                    # the exact regime (n < nout). No attention bias to apply.
                    log_key_weights = None
                else:
                    _log_key_weights = _log_key_weights.transpose(1, 2)
                    assert _log_key_weights.device == compressed_keys.device
                    # Compute attention bias using pad (avoids zeros + slice assignment + CPU->GPU transfer)
                    log_key_weights = torch.nn.functional.pad(
                        _log_key_weights,
                        (self.n_sink, self.n_window),
                        mode='constant',
                        value=0.0
                    )
                    assert log_key_weights.device == compressed_keys.device

                # Build key/value lists and do single cat (avoids multiple intermediate tensors)
                key_parts = []
                value_parts = []
                if self.n_sink > 0:
                    key_parts.append(keys[:, :, :self.n_sink])
                    value_parts.append(values[:, :, :self.n_sink])
                key_parts.append(compressed_keys)
                value_parts.append(compressed_values)
                if self.n_window > 0:
                    key_parts.append(keys[:, :, -self.n_window:])
                    value_parts.append(values[:, :, -self.n_window:])

                if len(key_parts) > 1:
                    compressed_keys = torch.cat(key_parts, dim=2)
                    compressed_values = torch.cat(value_parts, dim=2)

        return compressed_keys, compressed_values, log_key_weights

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Forward hook that manages decoding-specific compression logic.

        This hook:
        1. Detects when we're in decoding phase (not prefilling)
        2. Accumulates hidden states in a buffer
        3. Applies compression every N steps
        4. Clears the buffer after compression
        """
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx

        # Only operate during decoding phase (after prefilling)
        if kwargs["cache_position"][-1] <= q_len:
            # We're still in prefilling phase, don't do anything
            return output
        if layer_idx not in self.express_states:
            # target_size overloads h
            h = self.target_size
            self.express_states[layer_idx] = ExpressModule(
                nout=2 ** h,
                max_halving_rounds=2 * (h // 2),
                softmax_temp=module.scaling,
                alias=str(layer_idx),
                use_balancekv=self.use_balancekv,
            )
        if layer_idx not in self.cache_sizes:
            self.cache_sizes[layer_idx] = []
        # Add current hidden states to buffer for this layer
        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())

        self.layer_step_counts[layer_idx] += 1

        # NOTE: compression_interval is still the DecodingPress default up until this point
        bucket_size = self.express_states[layer_idx].get_bucket_size()
        compression_interval = bucket_size - (self.express_states[layer_idx].n % bucket_size)
        # Apply compression if we've reached the compression step threshold
        if (self.layer_step_counts[layer_idx] >= compression_interval):
            logger.debug(
                f"Applying decoding compression: layer_step_count ({self.layer_step_counts[layer_idx]}) >= compression_steps ({compression_interval})"  # noqa: E501
            )

            cache_layer = cache.layers[module.layer_idx]
            keys, values = extract_keys_and_values(cache, module.layer_idx)
            # if layer_idx == 0:
            #     import pdb; pdb.set_trace()

            # Get attention weights from output
            attentions = output[1] if len(output) > 1 and output[1] is not None else None

            # Apply compression using buffered hidden states for this layer
            buffered_hidden_states = torch.cat(self.hidden_states_buffer[layer_idx], dim=1)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            compress_output = self.compress(module, buffered_hidden_states, keys, values, attentions, kwargs)
            end.record()
            self.events[str(layer_idx)].append((start, end))
            if len(compress_output) == 2:
                keys, values = compress_output
            else:
                assert len(compress_output) == 3, "compress method must return either 2 or 3 tensors"
                keys, values, log_key_weights = compress_output
                if log_key_weights is not None:
                    cache_layer._log_key_weights = log_key_weights
            logger.debug(f"Applied decoding compression: " f"keys.shape: {keys.shape}, values.shape: {values.shape}")

            # Update cache with compressed keys and values
            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
                cache_layer.cumulative_length = keys.shape[2]
            else:
                cache_layer.keys = keys
                cache_layer.values = values

            # Reset step count and clear buffer for this layer
            self.layer_step_counts[layer_idx] = 0
            # Always clear the buffer after compression - otherwise there's a mismatch between
            # hidden states buffer and kv cache
            self.hidden_states_buffer[layer_idx] = []

        self.hidden_states_buffer[layer_idx] = (
            self.hidden_states_buffer[layer_idx][-self.hidden_states_buffer_size :]
            if self.hidden_states_buffer_size > 0
            else []
        )

        # update cache size at the current step
        self.cache_sizes[layer_idx].append(cache.layers[layer_idx].keys.shape[2])
        return output

    def reset(self):
        """Reset the decoding press state."""
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)

        # Reset Express
        self.express_states = {}
        self.cache_sizes = {}

        self.events = defaultdict(list)
