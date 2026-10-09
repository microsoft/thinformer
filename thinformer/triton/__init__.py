"""Triton implementation of Thinformer Attention."""

# Standard imports
import torch
from torch import nn
import math
import logging

from ..timing_utils import multi_events
from ..generic_utils import log4_largest_power_of_four
from .khcompress import khcompress
from .flash_attn import attention

# Thinformer Express imports
from .offline_express import thinformer_express

logger = logging.getLogger(__name__)


class ThinformerAttention(nn.Module):
    """Implementation of Thinformer Attention module."""

    def __init__(
        self,
        h: int = 10,
        scale: float | None = None,
        delta: float = 0.5,
        num_key_value_groups: int = 1,
        n_first: int = 0,
        n_last: int = 0,
        alias: str = "thinformer",
        **kwargs: dict,
    ):
        """Initialize the ThinformerAttention module.

        Args:
            h (int): log2 target cache size
            scale (float): scale for dot-product attention.
              If `None`, scale is chosen as 1/sqrt(key.shape[-1]) in forward.
            delta (float): the delta parameter for the attention mechanism
            num_key_value_groups (int): the number of key-value groups
            n_first (int): the number of attention sinks
            n_last (int): the size of the sliding window
            max_halving_rounds (int | None): the maximum number of halving rounds
            alias (str): alias for profiling event names
            kwargs: placeholder for other arguments

        """
        # import pdb; pdb.set_trace()
        assert isinstance(h, int) and h >= 0, (
            f"h must be a nonnegative integer, got {h}"
        )
        super().__init__()
        self.h = h
        self.scale = scale
        self.alias = alias
        self.delta = delta
        self.num_key_value_groups = num_key_value_groups
        self.n_first = n_first  # number of attention sinks
        self.n_last = n_last  # size of sliding window

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """When causal=False, use the Thinformer algorithm, which comprises
        two steps:
        1. Select a subset of the key and value pairs using KH-Compress.
           The size of this subset is 2^h.
        2. For each query vector, only attend to the selected key and value
           pairs. The computational complexity of this step is 2^h * T.

        When causal=True, use the Thinformer Express algorithm, which comprises
        two steps:
        1. Select a subset of the key and value pairs using KH-Express.
           Each query has a different subset with size O(2^h).
        2. For each query vector, only attend to the selected key and value
           pairs. The computational complexity of this step is O(2^h * T).
        To improve GPU utilization, we interleave steps 1 and 2 across queries.

        Args:
            query: (B, T, H, E) The tensor containing the queries
            key: (B, S, H, E) The tensor containing the keys
            value: (B, S, H, D) The tensor containing the values
            causal: (bool) If True, S must equal T

        Returns:
            output: (B, T, H, D) The tensor approximating
                softmax(query * tranpose(key) * softmax_temp * mask) * value
                for softmax_temp = self.scale or 1 / sqrt(E)
            A: None. Placeholder to ensure compatibility with downstream applications
                that expect a second return value.

        """
        B, S, H, E = key.shape
        _, _, _, D = value.shape
        softmax_temp = self.scale or 1 / math.sqrt(E)

        if causal:
            assert query.shape[1] == S, (
                "Causal attention requires query and key to have the same sequence length"
            )
            # Default max_halving_rounds sets the smallest Halve input size to be 4^2,
            # the minimum block size in Triton. h < 4 yields 0: no compress
            # rounds at all (every round runs fully subsampled, mstar == 0).
            max_halving_rounds = max(0, 2 * (self.h // 2 - 1))
            out = thinformer_express(
                query=query,
                key=key,
                value=value,
                nout=2**self.h,
                max_halving_rounds=max_halving_rounds,
                softmax_temp=softmax_temp,
                delta=self.delta,
                n_first=self.n_first,
                n_last=self.n_last,
                num_key_value_groups=self.num_key_value_groups,
                alias=self.alias,
            )
        else:
            if self.n_first > 0 or self.n_last > 0:
                logger.warning(
                    f"n_first={self.n_first} and n_last={self.n_last} are ignored for non-causal attention"
                )
            # Compute squareroot of softmax temperature
            # for the attention that A = softmax(query * key * softmax_temp)
            sqrt_softmax_temp = math.sqrt(softmax_temp)
            log4_n = log4_largest_power_of_four(S)
            # 2^h = 2^g * sqrt(largest power of four less than or equal to n) = 2^log4_n
            g = self.h - log4_n
            logger.info(f"Setting g={g} for log4_n={log4_n} and h={self.h}")

            # Thinning
            with multi_events(f"{self.alias}__thinning"):
                key, value = khcompress(
                    key * sqrt_softmax_temp,
                    value,
                    log4_n,
                    g=g,
                    alias=self.alias,
                )
                key = key * sqrt_softmax_temp
                sm_scale = 1.0

            # Final attention computation
            with multi_events(f"{self.alias}__attention"):
                out = attention(
                    q=query.transpose(1, 2),
                    k=key.transpose(1, 2),
                    v=value.transpose(1, 2),
                    causal=False,
                    sm_scale=sm_scale,
                ).transpose(1, 2)

        return out, None
