"""Torch reference halve() (from 5a91452^) used by tests/test_block_halve.py."""

import math

import torch
import triton


def is_hip_cdna2() -> bool:
    """Return True when the active Triton target is an AMD CDNA2 (gfx90a) GPU."""
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx90a"


def naive_batch_exp_kernel(
    a_key: torch.Tensor,
    b_key: torch.Tensor,
    a_value: torch.Tensor,
    b_value: torch.Tensor,
    shift: torch.Tensor,
    b_sqd: torch.Tensor,
    softmax_temp: float,
    vbar: torch.Tensor = None,
    kbar: torch.Tensor = None,
) -> torch.Tensor:
    """Returns tensor of pairwise-differenced weighted exponential kernel matrices
    kernel_mat
        = exp(a_key @ b_key - shift)
        * (a_value @ b_value + b_sqd)
    Kdiff = kernel_mat[:, ::2, ::2] - kernel_mat[:, ::2, 1::2]
        - kernel_mat[:, 1::2, ::2] + kernel_mat[:, 1::2, 1::2]

    Note: Assumes key has already been scaled appropriately by
    sqrt(softmax_temp)

    Args:
        a_key: tensor of shape [B, M, E]
        b_key: tensor of shape [B, E, N]
        a_value: tensor of shape [B, M, D]
        b_value: tensor of shape [B, D, M]
        shift: tensor broadcastable to shape [B, 1, 1]
        b_sqd: tensor of shape [B, 1, 1]
        softmax_temp: scalar, softmax temperature for the exp kernel
        vbar: tensor of shape [B, 1, D]
        kbar: tensor of shape [B, 1, E]

    Returns tensor of shape [B, M // 2, N // 2]

    """
    if kbar is None:
        a_key_minus_kbar = a_key
        b_key_minus_kbar = b_key
    else:
        a_key_minus_kbar = a_key - kbar
        b_key_minus_kbar = b_key - kbar.mT
    key_term = torch.einsum(
        "bme,ben->bmn", a_key_minus_kbar.float(), b_key_minus_kbar.float()
    )
    key_term = key_term - shift
    key_term = torch.exp_(key_term * softmax_temp)
    if vbar is None:
        a_value_minus_vbar = a_value
        b_value_minus_vbar = b_value
    else:
        a_value_minus_vbar = a_value - vbar
        b_value_minus_vbar = b_value - vbar.mT
    val_term = torch.einsum(
        "bmd,bdn->bmn", a_value_minus_vbar.float(), b_value_minus_vbar.float()
    )
    val_term = val_term + b_sqd
    val_term = val_term * key_term
    val_term = val_term.to(a_key.dtype)

    Kdiff = val_term[..., ::2] - val_term[..., 1::2]
    Kdiff = Kdiff[:, ::2] - Kdiff[:, 1::2]
    return Kdiff


def halve(
    a_key: torch.Tensor,
    b_key: torch.Tensor,
    a_value: torch.Tensor,
    b_value: torch.Tensor,
    halve_prob: float,
    shift: torch.Tensor,
    b_sqd: torch.Tensor,
    softmax_temp: float = 1.0,
    vbar: torch.Tensor = None,
    kbar: torch.Tensor = None,
    uniforms_buf: torch.Tensor = None,
    symmetrize_rv: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequential kernel halving of each batch element.

    Args:
        a_key: tensor of shape [B, M, E]
        b_key: tensor of shape [B, E, M]
        a_value: tensor of shape [B, M, D]
        b_value: tensor of shape [B, D, M]
        halve_prob: used to compute delta = halve_prob * M^2
        shift: tensor of shape [B, 1, 1] accepted by naive_batch_exp_kernel;
            typically represents the maximum squared key vector two norm
            to prevent exponential overflow via log-sum-exp trick
        b_sqd: tensor of shape [B, 1, 1] accepted by naive_batch_exp_kernel;
            typically represents the maximum squared value vector inf norm
        softmax_temp: scalar, softmax temperature for the exp kernel
        vbar: tensor of shape [B, 1, D]
        kbar: tensor of shape [B, 1, E]
        uniforms_buf: pre-allocated uniform random variables
        symmetrize_rv: if given, a bool tensor XORed with the swap decisions to
            choose between the coreset and its complement

    Returns:
        Kdiff: tensor of shape [B, M // 2, M // 2]
        coreset_indices: tensor of shape [B, M // 2]
        o_key: tensor of shape [B, M // 2, E]
        o_value: tensor of shape [B, M // 2, D]

    """
    assert a_key.shape[2] == b_key.shape[1], "Incompatible dimensions"
    assert a_value.shape[2] == b_value.shape[1], "Incompatible dimensions"
    B, M, E = a_key.shape
    B, M, D = a_value.shape
    assert M % 2 == 0, "M must be even"

    num_points_in_coreset = M // 2
    if uniforms_buf is None:
        delta = halve_prob * M * M
        log_multiplier = 0.5 + math.log(2 * M / delta)
        uniforms = torch.empty(B, num_points_in_coreset, device=a_key.device).uniform_(
            -log_multiplier, log_multiplier
        )
    else:
        uniforms = uniforms_buf

    shift = torch.repeat_interleave(shift, B // shift.shape[0], dim=0)
    b_sqd = torch.repeat_interleave(b_sqd, B // b_sqd.shape[0], dim=0)

    Kdiff = naive_batch_exp_kernel(
        a_key, b_key, a_value, b_value, shift, b_sqd, softmax_temp, vbar, kbar
    )

    rt_diag = torch.sqrt(Kdiff.diagonal(dim1=-2, dim2=-1))
    uniforms = uniforms * rt_diag * torch.cummax(rt_diag, dim=-1).values

    swap_points = torch.zeros(
        B, num_points_in_coreset, device=Kdiff.device, dtype=torch.bool
    )
    # The 0-th point never swaps, so coreset_sum_diff starts as the 0-th row of Kdiff
    coreset_sum_diff = Kdiff[..., 0, :]
    for i in range(1, num_points_in_coreset):
        swap_points_i = uniforms[..., i] <= coreset_sum_diff[..., i]
        swap_points[..., i] = swap_points_i
        coreset_sum_diff = coreset_sum_diff + (
            1.0 - 2.0 * swap_points_i.unsqueeze(-1)
        ) * Kdiff.select(-2, i)
    if symmetrize_rv is not None:
        swap_points = torch.logical_xor(swap_points, symmetrize_rv)
    coreset_indices = swap_points + torch.arange(
        0, M, step=2, device=Kdiff.device, dtype=torch.int32
    ).expand_as(swap_points)

    coreset = coreset_indices.unsqueeze(-1).to(torch.int64)
    o_key = a_key.gather(1, coreset.expand(B, M // 2, E))
    o_value = a_value.gather(1, coreset.expand(B, M // 2, D))
    return Kdiff, coreset_indices, o_key, o_value
