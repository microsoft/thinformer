"""Triton implementation of Kernel Halving."""

import os
import torch

from .block_halve import block_halve as triton_halve
from .balanced_walk import balanced_walk
from ..timing_utils import multi_events

M_INNER = int(os.environ.get("M_INNER", "64"))


def halve_with_centering(
    key: torch.Tensor,
    value: torch.Tensor,
    halve_prob: float,
    symmetrize: bool = True,
    softmax_temp: float = 1.0,
    alias: str = "halve",
    b_sqd: torch.Tensor | None = None,
    uniforms: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel Halving with key and value centering

    NOTE: the only way to ensure reproducibility with this implementation is to set the random seed.

    Args:
        key: tensor of shape [B, S, H, E]
        value: tensor of shape [B, S, H, D]
        halve_prob: halve_prob * S^2 is used to compute the delta for halve_K
        symmetrize: if True, randomly choose between coreset and its complement
        softmax_temp: softmax temperature for the exp kernel
        alias: alias for the event
        b_sqd: tensor of shape [B, 1, H, 1], b_sqd parameter for the exp kernel
        uniforms: tensor of shape [B, 1, S // 2], buffer of uniform RVs

    Returns:
        compressed_key: tensor of shape [B, S//2, H, E]
        compressed_value: tensor of shape [B, S//2, H, D]

    """
    assert symmetrize is True
    with multi_events(f"{alias}__halve"):
        with multi_events(f"{alias}__halve__prologue1"):
            # compute vector for centering the keys
            kbar = torch.mean(key, dim=1, keepdim=True)
            # compute vector for centering the values
            vmax, _ = torch.max(value, dim=1, keepdim=True)
            vmin, _ = torch.min(value, dim=1, keepdim=True)
            vbar = 0.5 * (vmax + vmin)

            if b_sqd is None:
                # Compute the b_sqd parameter
                # NOTE: we can set b_sqd on a per-halving problem basis without sacrificing error
                b_sqd = (
                    torch.linalg.vector_norm(
                        value - vbar, dim=(1, 3), ord=torch.inf, keepdim=True
                    )
                    ** 2
                )
            # import pdb; pdb.set_trace()
            B, n, H, E = key.shape
            _, _, _, D = value.shape
        # NOTE: the current Triton implementation of halve cannot handle problems with less than M_INNER points
        # TODO: implement padding for _batch_halve_inner
        with multi_events(f"{alias}__halve__prologue2"):
            input_indices = torch.arange(0, n, device=key.device, dtype=torch.int32)[
                None, None, :
            ].expand(B * H, 1, n)
        with multi_events(f"{alias}__halve__triton"):
            _, _, coreset_indices = triton_halve(
                (key - kbar).transpose(1, 2).view(B * H, n, E),
                (key - kbar).transpose(1, 2).view(B * H, n, E).mT,
                (value - vbar).transpose(1, 2).view(B * H, n, D),
                (value - vbar).transpose(1, 2).view(B * H, n, D).mT,
                halve_prob=halve_prob,
                b_sqd=b_sqd.transpose(1, 2).view(B * H, 1, 1),
                M_inner=M_INNER,
                input_indices=input_indices,
                softmax_temp=softmax_temp,
                symmetrize=symmetrize,
                uniforms_buf=uniforms,
            )
        with multi_events(f"{alias}__halve__gather"):
            compressed_key = (
                key.transpose(1, 2)
                .view(B * H, n, E)
                .gather(
                    dim=1,
                    index=coreset_indices.squeeze(1)
                    .unsqueeze(-1)
                    .expand(-1, -1, E)
                    .to(torch.int64),
                )
                .view(B, H, -1, E)
                .transpose(1, 2)
            )
            compressed_value = (
                value.transpose(1, 2)
                .view(B * H, n, D)
                .gather(
                    dim=1,
                    index=coreset_indices.squeeze(1)
                    .unsqueeze(-1)
                    .expand(-1, -1, D)
                    .to(torch.int64),
                )
                .view(B, H, -1, D)
                .transpose(1, 2)
            )
    return compressed_key, compressed_value


def halve_balancekv(
    key: torch.Tensor,
    value: torch.Tensor,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Halving with balanced_walk instead of Kernel Halving

    Args:
        key: tensor of shape [B, S, H, E]
        value: tensor of shape [B, S, H, D]
        halve_prob: halve_prob * S^2 is used to compute the delta for halve_K
        symmetrize: if True, randomly choose between coreset and its complement
        softmax_temp: softmax temperature for the exp kernel
        alias: alias for the event
        b_sqd: tensor of shape [B, 1, H, 1], b_sqd parameter for the exp kernel
        uniforms: tensor of shape [B, 1, S // 2], buffer of uniform RVs
        verbose: if True, print debug information

    Returns:
        compressed_key: tensor of shape [B, S//2, H, E]
        compressed_value: tensor of shape [B, S//2, H, D]

    """
    block_size: int = 128
    beta: float = 0.0
    temp: float = 1.0
    gamma: float = 4.0
    itrs = 1
    # TODO (Albert): is it still deterministic since we fix the seed in the calling code
    rng = None
    # if verbose:
    #     import pdb; pdb.set_trace()
    indices, _ = balanced_walk(
        key.transpose(1, 2),
        rng,
        gamma,
        temp,
        beta,
        itrs,
        block_size,
        value=value.transpose(1, 2),
    )
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1]).transpose(1, 2)
    compressed_key = key.gather(1, indices)
    compressed_value = value.gather(1, indices)
    return compressed_key, compressed_value
