# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Press for the BalanceKV algorithm."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.balanced_walk import balanced_walk
from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class BalanceKVPress(BasePress):
    target_size: float = 0.0  # fraction of keys and values to discard

    # BalanceKV hyperparameters (see https://github.com/microsoft/express/blob/353bc2ffb027081e91368b785f52cef789d6534a/examples/balancekv/run_longbench.py#L175)
    sink_size: int = 32
    window_size: int = 32
    block_size: int = 128
    beta: float = 0.0
    temp: float = 1.0
    gamma: float = 4.0
    rng: torch.Generator = torch.Generator().manual_seed(1)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress the keys and values using the BalanceKV algorithm.

        Args:
            module (nn.Module): The attention module.
            hidden_states (torch.Tensor): The hidden states of the model
            keys (torch.Tensor): The keys from the attention mechanism of shape (b, h, n, d)
            values (torch.Tensor): The values from the attention mechanism of shape (b, h, n, d)
            attentions (torch.Tensor): The attention weights of shape (b, h, n) or None
            kwargs (dict): Additional arguments, including position embeddings.

        Returns:
            compressed_keys (torch.Tensor): The compressed keys of shape (b, h, cache_size, d)
            compressed_values (torch.Tensor): The compressed values of shape (b, h, cache_size, d)
            log_key_weights (torch.Tensor): Log-weights applied to the kept keys, or None
        """
        B, H, S, E = keys.shape
        self.seq_len = S
        if (self.target_size == 0) or (self.target_size >= S - self.sink_size - self.window_size):
            # If the sequence length is less than or equal to the cache size, return keys and values as is
            key_quant = keys
            val_quant = values
            log_key_weights = None
        else:
            itrs = math.log2(1 / (1 - self.target_size))
            assert itrs == int(itrs), f"target_size must be a power of 2, but got {self.target_size}"
            itrs = int(itrs)

            k_compressed = keys[:, :, self.sink_size:-self.window_size]
            v_compressed = values[:, :, self.sink_size:-self.window_size]
            indices, weights = balanced_walk(
                k_compressed, self.rng, self.gamma, self.temp, self.beta, itrs, self.block_size, value=v_compressed
            )
            key_quant = k_compressed.gather(
                dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
            )
            val_quant = v_compressed.gather(
                dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
            )
            weights = weights / 2 ** (itrs)
            weights = weights.unsqueeze(-1)
            val_quant = (val_quant * weights).to(values.dtype)

            # NOTE: instead of repeating the compressed keys and values, we can update attn_mask
            log_key_weights = torch.zeros(
                (B, H, key_quant.shape[-2] + self.sink_size + self.window_size), device=keys.device, dtype=keys.dtype
            )
            # Upweight the keys by nin/nout
            log_key_weights[:, :, self.sink_size:-self.window_size] = math.log(
                S - self.sink_size - self.window_size
            ) - math.log(key_quant.shape[-2])

            if self.sink_size > 0:
                # Concatenate the first sink_size keys and values
                key_quant = torch.cat((keys[:, :, :self.sink_size], key_quant), dim=2)
                val_quant = torch.cat((values[:, :, :self.sink_size], val_quant), dim=2)
            if self.window_size > 0:
                # Concatenate the last window_size keys and values
                key_quant = torch.cat((key_quant, keys[:, :, -self.window_size:]), dim=2)
                val_quant = torch.cat((val_quant, values[:, :, -self.window_size:]), dim=2)
        return key_quant, val_quant, log_key_weights


@dataclass
class BalanceKVScorerPress(ScorerPress):
    """
    ScorerPress variant of BalanceKV, usable standalone or wrapped in DecodingPress.

    Keeps the first n_sink and last n_window tokens, runs the balanced walk over the
    middle tokens, and keeps exactly the walk-selected half (per halving iteration).
    compress() is overridden, so the output size is
    n_sink + n_window + (k_len - n_sink - n_window) // 2**itrs, NOT k_len * (1 -
    compression_ratio). The balanced-walk value weights and the log_key_weights
    upweighting of BalanceKVPress are intentionally dropped.

    1 - compression_ratio must be an inverse power of two (0.5 -> 1 halving,
    0.75 -> 2, ...). When wrapped in DecodingPress, use size_based_trigger=True and
    compression_interval == target_size; target_size counts middle tokens only
    (DecodingPress adds n_sink + n_window to its trigger threshold).
    """

    n_sink: int = 32
    n_window: int = 32
    block_size: int = 128
    beta: float = 0.0
    temp: float = 1.0
    gamma: float = 4.0
    rng: torch.Generator = torch.Generator().manual_seed(1)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        itrs = math.log2(1 / (1 - self.compression_ratio))
        assert itrs == int(itrs), (
            f"1 - compression_ratio must be an inverse power of two, got compression_ratio={self.compression_ratio}. "
            "When wrapped in DecodingPress, set size_based_trigger=True and compression_interval == target_size "
            "so compression is always triggered with a middle-token count of exactly 2 * target_size."
        )

        k_len = keys.shape[2]
        if k_len <= self.n_sink + self.n_window:
            return keys, values

        mid_keys = keys[:, :, self.n_sink : k_len - self.n_window]
        mid_values = values[:, :, self.n_sink : k_len - self.n_window]
        indices, _ = balanced_walk(
            mid_keys, self.rng, self.gamma, self.temp, self.beta, int(itrs), self.block_size, value=mid_values
        )
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        kept_keys = mid_keys.gather(2, indices)
        kept_values = mid_values.gather(2, indices)

        if self.n_sink > 0:
            kept_keys = torch.cat((keys[:, :, : self.n_sink], kept_keys), dim=2)
            kept_values = torch.cat((values[:, :, : self.n_sink], kept_values), dim=2)
        if self.n_window > 0:
            kept_keys = torch.cat((kept_keys, keys[:, :, -self.n_window :]), dim=2)
            kept_values = torch.cat((kept_values, values[:, :, -self.n_window :]), dim=2)
        return kept_keys.contiguous(), kept_values.contiguous()
