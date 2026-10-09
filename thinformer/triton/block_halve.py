"""Triton kernel halving.

Consolidates the former exp_kernel.py and halve.py into this file: keeps only
the block_halve() code path and its transitive triton-jit dependencies
(_exp_kernel_single_block, _exp_kernel_single_block_2, _halve_K,
_batch_halve_inner, _batch_flash_psi).
"""

import triton
import triton.language as tl

import torch
import math


MIN_BLOCK_SIZE = 16  # minimum block size for tl.dot


@triton.jit
def _exp_kernel_single_block(
    # Pointers to matrices
    a_key,
    b_key,
    a_value,
    b_value,
    shift,
    b_sqd_ptr,
    softmax_temp: tl.constexpr,
    INIT: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  #
):
    """Kernel for computing the Kdiff matrix (C) on a single block, defined as
    K = exp(A_key x B_key - shift) * (A_value x B_value + b_sqd)
    Kdiff = K[:, ::2] - K[:, 1::2]
    Kdiff = Kdiff[::2] - Kdiff[1::2]

    NOTE: shift is updated in an online manner.

    Args:
        a_key: tensor of shape [BLOCK_SIZE_M, E]
        b_key: tensor of shape [E, BLOCK_SIZE_M]
        a_value: tensor of shape [BLOCK_SIZE_M, D]
        b_value: tensor of shape [D, BLOCK_SIZE_M]
        shift: tensor of shape [BLOCK_SIZE_M // 2]
        b_sqd_ptr: tensor of shape [BLOCK_SIZE_M]
        softmax_temp: scalar, softmax temperature for the exp kernel

    Returns:
        Kdiff: tensor of shape [BLOCK_SIZE_M//2, BLOCK_SIZE_M//2]
        new_shift: tensor of shape [BLOCK_SIZE_M]

    """
    # -----------------------------------------------------------
    # Load the b_sqd value.
    # NOTE: we cannot compute b_sqd from the current block alone
    b_sqd = tl.load(b_sqd_ptr)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    key_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    key_accumulator = tl.dot(a_key, b_key * softmax_temp, key_accumulator)
    # Update shift
    row_max = tl.max(key_accumulator, axis=1)
    pair_max = tl.max(row_max.reshape(BLOCK_SIZE_M // 2, 2), axis=1, keep_dims=True)
    if INIT:
        # NOTE: overwrite (not `*= 0`) -- the buffer is torch.empty and may
        # contain NaN/Inf garbage, and 0 * NaN = NaN would poison the state.
        shift = tl.zeros_like(shift)
    new_shift = tl.maximum(shift.expand_dims(1), pair_max)
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    # NOTE: triton_output and torch_output differ absolutely by 1e-2 when introducing the shift term
    key_accumulator = key_accumulator - new_shift.broadcast_to(
        BLOCK_SIZE_M // 2, 2
    ).reshape(BLOCK_SIZE_M, 1)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    # key_accumulator = tl.exp(key_accumulator)
    # Can also use exp2
    key_accumulator = tl.exp2(key_accumulator * 1.44269504)

    value_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    value_accumulator = tl.dot(a_value, b_value, value_accumulator)
    value_accumulator = value_accumulator + b_sqd
    # Compute the kernel matrix
    K = key_accumulator * value_accumulator

    # -----------------------------------------------------------
    # Compute the Kdiff matrix while still in fp32
    K_even, K_odd = K.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2).split()
    Kdiff = K_even - K_odd
    Kdiff_even, Kdiff_odd = (
        Kdiff.trans(1, 0).reshape(BLOCK_SIZE_N // 2, BLOCK_SIZE_M // 2, 2).split()
    )
    Kdiff = (Kdiff_even - Kdiff_odd).trans(1, 0)
    # Return Kdiff and new_shift as a flattened tensor
    return Kdiff, tl.ravel(new_shift)


@triton.jit
def _exp_kernel_single_block_2(
    # Pointers to matrices
    key_ptr,
    value_ptr,
    indices,
    shift,
    b_sqd_ptr,
    softmax_temp: tl.constexpr,
    E,
    D,
    INIT: tl.constexpr,
    stride_key_m,
    stride_key_e,
    stride_value_m,
    stride_value_d,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_E: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,  #
):
    """Kernel for computing the Kdiff matrix (C) on a single block, defined as
    K = exp(A_key x B_key - shift) * (A_value x B_value + b_sqd)
    Kdiff = K[:, ::2] - K[:, 1::2]
    Kdiff = Kdiff[::2] - Kdiff[1::2]

    NOTE: shift is updated in an online manner.

    Args:
        a_key: tensor of shape [BLOCK_SIZE_M, E]
        b_key: tensor of shape [E, BLOCK_SIZE_M]
        a_value: tensor of shape [BLOCK_SIZE_M, D]
        b_value: tensor of shape [D, BLOCK_SIZE_M]
        shift: tensor of shape [BLOCK_SIZE_M // 2]
        b_sqd_ptr: tensor of shape [BLOCK_SIZE_M]
        softmax_temp: scalar, softmax temperature for the exp kernel

    Returns:
        Kdiff: tensor of shape [BLOCK_SIZE_M//2, BLOCK_SIZE_M//2]
        new_shift: tensor of shape [BLOCK_SIZE_M]

    """
    # -----------------------------------------------------------
    # Load the b_sqd value.
    # NOTE: we cannot compute b_sqd from the current block alone
    b_sqd = tl.load(b_sqd_ptr)
    offs_e = tl.arange(0, BLOCK_SIZE_E)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    key_ptrs = key_ptr + (
        indices[:, None] * stride_key_m + offs_e[None, :] * stride_key_e
    )
    value_ptrs = value_ptr + (
        indices[:, None] * stride_value_m + offs_d[None, :] * stride_value_d
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    # -- Compute KK^T --
    key_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(E, BLOCK_SIZE_E)):
        key = tl.load(key_ptrs, mask=offs_e[None, :] < E - k * BLOCK_SIZE_E, other=0.0)
        # We accumulate along the E dimension.
        key_accumulator = tl.dot(key, key.T * softmax_temp, key_accumulator)
        # Advance the ptrs to the next E block.
        key_ptrs += BLOCK_SIZE_E * stride_key_e
    # -- Compute exp(KK^T - shift) --
    # Update shift
    row_max = tl.max(key_accumulator, axis=1)
    pair_max = tl.max(row_max.reshape(BLOCK_SIZE_M // 2, 2), axis=1, keep_dims=True)
    if INIT:
        # NOTE: overwrite (not `*= 0`) -- the buffer is torch.empty and may
        # contain NaN/Inf garbage, and 0 * NaN = NaN would poison the state.
        shift = tl.zeros_like(shift)
    new_shift = tl.maximum(shift.expand_dims(1), pair_max)
    key_accumulator = key_accumulator - new_shift.broadcast_to(
        BLOCK_SIZE_M // 2, 2
    ).reshape(BLOCK_SIZE_M, 1)
    key_accumulator = tl.exp2(key_accumulator * 1.44269504)
    # -- Compute VV^T --
    value_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        value = tl.load(
            value_ptrs, mask=offs_d[None, :] < D - k * BLOCK_SIZE_D, other=0.0
        )
        # We accumulate along the D dimension.
        value_accumulator = tl.dot(value, value.T, value_accumulator)
        # Advance the ptrs to the next D block.
        value_ptrs += BLOCK_SIZE_D * stride_value_d
    value_accumulator = value_accumulator + b_sqd
    # -- Compute the kernel matrix --
    K = key_accumulator * value_accumulator

    # -----------------------------------------------------------
    # Compute the Kdiff matrix while still in fp32
    K_even, K_odd = K.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2).split()
    Kdiff = K_even - K_odd
    Kdiff_even, Kdiff_odd = (
        Kdiff.trans(1, 0).reshape(BLOCK_SIZE_N // 2, BLOCK_SIZE_M // 2, 2).split()
    )
    Kdiff = (Kdiff_even - Kdiff_odd).trans(1, 0)
    # Return Kdiff and new_shift as a flattened tensor
    return Kdiff, tl.ravel(new_shift)


@triton.jit
def _halve_K(
    shift_ptr,
    Kdiff_ptr,
    uniform_ptr,
    coreset_sum_diff_ptr,
    rt_diag_cummax,
    swap_points_ptr,
    coreset_sum_diff,
    M: tl.constexpr,
    stride_shift_m,
    stride_Kdiff_m,
    stride_Kdiff_n,
    stride_uniform_m,
    stride_coreset_sum_diff_m,
    stride_swap_points_m,
    BLOCK_SIZE_M: tl.constexpr,
    INIT: tl.constexpr,
):
    """Triton kernel for halving the input points given Kdiff.

    Args:
        a_key_ptr: pointer to the input key matrix
        a_value_ptr: pointer to the input value matrix
        shift: tensor of shape [M] used when computing coreset_sum_diff
        new_shift: tensor of shape [M] from most recent exp_kernel call
        Kdiff_ptr: pointer to the Kdiff matrix
        uniform_ptr: pointer to the uniform matrix
        coreset_sum_diff_ptr: pointer to the coreset_sum_diff matrix
        rt_diag_cummax: rt_diag_cummax scalar
        eta_ptr: pointer to the eta matrix
        INIT: if True, initialize 0-th swap point to 0
        OFFSET: offset to add to the coreset indices

    Returns:
        rt_diag_cummax: rt_diag_cummax scalar

    """
    # pdb.set_trace()
    # ------------------------------------------------------------
    # Create pointers to the 0-th row of Kdiff
    offs_Kdiff_n = tl.arange(0, BLOCK_SIZE_M // 2)
    Kdiff_row_ptrs = Kdiff_ptr + (offs_Kdiff_n[None, :] * stride_Kdiff_n)
    # Get pointers to coreset_sum_diff accumulator
    offs_coreset_sum_diff_m = tl.arange(0, BLOCK_SIZE_M // 2)
    coreset_sum_diff_ptrs = coreset_sum_diff_ptr + (
        offs_coreset_sum_diff_m[None, :] * stride_coreset_sum_diff_m
    )
    # ------------------------------------------------------------
    # Initialize rt_diag_cummax
    # rt_diag_cummax = tl.load(rt_diag_cummax_ptr)
    # ------------------------------------------------------------
    # Iterate over the rows of Kdiff
    for i in tl.range(0, M // 2, num_stages=4):
        # pdb.set_trace()
        #  -- compute swap point --
        # Compute the square root of the i-th diagonal element of Kdiff at position (i, i)
        rt_diag = tl.sqrt(
            tl.load(Kdiff_ptr + i * stride_Kdiff_m + i * stride_Kdiff_n).to(tl.float32)
        )
        # rescale rt_diag_cummax by exp(0.5 * (new_shift[i-1] - new_shift[i]))
        # so that it uses the same shift as rt_diag
        # NOTE: we use a mask to avoid illegal memory access when i=0 and INIT=True
        scale = tl.where(
            (i == 0) and INIT,
            1,
            tl.exp(
                0.5
                * (
                    tl.load(
                        shift_ptr + (i - 1) * stride_shift_m,
                        mask=(i > 0) | (not INIT),
                        other=0.0,
                    )
                    - tl.load(shift_ptr + i * stride_shift_m)
                ).to(tl.float32)
            ),
        )
        # Update the cumulative maximum of the diagonal elements
        rt_diag_cummax = tl.maximum(rt_diag, rt_diag_cummax * scale)
        # Compute scaled uniform
        uniform = tl.load(uniform_ptr + i * stride_uniform_m) * rt_diag * rt_diag_cummax
        # Flip a biased coin to determine swap point
        swap_point = tl.where(
            (i == 0) and INIT,
            0,
            uniform <= tl.load(coreset_sum_diff_ptr + i * stride_coreset_sum_diff_m),
        )
        # pdb.set_trace()
        #  -- update coreset_sum_diff and eta --
        # Load the i-th row of Kdiff
        Kdiff_row_i = tl.load(
            Kdiff_row_ptrs, mask=offs_Kdiff_n[None, :] < (M // 2), other=0.0
        )
        # Update coreset_sum_diff
        eta = 1 - 2 * swap_point
        coreset_sum_diff = coreset_sum_diff + eta * Kdiff_row_i
        # Write back coreset_sum_diff!!!
        tl.store(
            coreset_sum_diff_ptrs,
            coreset_sum_diff,
            mask=offs_coreset_sum_diff_m[None, :] < (M // 2),
        )
        # Write back the swap_points
        tl.store(swap_points_ptr + i * stride_swap_points_m, swap_point)
        # -- advance ptrs --
        # Advance the Kdiff pointer by one row
        Kdiff_row_ptrs += stride_Kdiff_m
    return rt_diag_cummax


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_E": e, "BLOCK_SIZE_D": d},
            num_stages=s,
            num_warps=w,
        )
        for e, d, s, w in [
            (64, 64, 3, 8),
            (32, 32, 4, 4),
            (32, 32, 5, 2),
            # Good configs for fp8 inputs.
            (128, 128, 3, 8),
            (128, 128, 4, 4),
            (64, 64, 4, 4),
        ]
    ],
    key=["E", "D"],
)
@triton.jit
def _batch_halve_inner(
    a_key_ptr,
    a_value_ptr,
    shift_ptr,
    b_sqd_ptr,
    softmax_temp: tl.constexpr,
    Kdiff_ptr,
    uniform_ptr,
    coreset_sum_diff_ptr,
    rt_diag_cummax_ptr,
    coreset_indices_ptr,
    input_indices_ptr,
    swap_points_ptr,
    symmetrize_rv_ptr,
    # Matrix dimensions
    A,
    M: tl.constexpr,
    E: tl.constexpr,
    D: tl.constexpr,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_a_key_b,
    stride_a_key_m,
    stride_a_key_e,
    stride_a_value_b,
    stride_a_value_m,
    stride_a_value_d,
    stride_shift_b,
    stride_shift_a,
    stride_shift_m,
    stride_b_sqd_b,
    stride_Kdiff_b,
    stride_Kdiff_a,
    stride_Kdiff_m,
    stride_Kdiff_n,
    stride_uniform_b,
    stride_uniform_a,
    stride_uniform_m,
    stride_coreset_sum_diff_b,
    stride_coreset_sum_diff_a,
    stride_coreset_sum_diff_m,
    stride_coreset_sum_diff_k,
    stride_rt_diag_cummax_b,
    stride_rt_diag_cummax_a,
    stride_coreset_indices_b,
    stride_coreset_indices_a,
    stride_coreset_indices_m,
    stride_input_indices_b,
    stride_input_indices_a,
    stride_input_indices_m,
    stride_swap_points_b,
    stride_swap_points_a,
    stride_swap_points_m,
    INIT: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_E: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Kernel for halving a single block of key, value points

    NOTE: A is the number of halving problems per batch and M is the size of each halving problem.

    Args:
        a_key_ptr: pointer to the key tensor A of shape [B, n, E]
        a_value_ptr: pointer to the value tensor A of shape [B, n, D]
        shift_ptr: pointer to the shift tensor of shape [B, A, M // 2]
        b_sqd_ptr: pointer to the b_sqd tensor of shape [B, 1, 1]
        input_indices_ptr: pointer to input indices of shape [B, A, M]
        softmax_temp: float, softmax temperature for the exp kernel
        Kdiff_ptr: pointer to the Kdiff tensor of shape [B, A, M // 2, M // 2]
        uniform_ptr: pointer to the uniform tensor of shape [B, A, M // 2]
        coreset_sum_diff_ptr: pointer to the coreset_sum_diff tensor of shape [B, A, M // 2, MIN_BLOCK_SIZE]
        rt_diag_cummax_ptr: pointer to the rt_diag_cummax tensor of shape [B, A]
        coreset_indices_ptr: pointer to the coreset indices tensor of shape [B, A, M // 2]
        swap_points_ptr: pointer to the swap points tensor of shape [B, A, M // 2]
        INIT: bool, if True, initialize the coreset_sum_diff and rt_diag_cummax tensors to 0

    """
    pid = tl.program_id(0)
    pid_b = pid // A
    pid_a = pid % A
    # print(f"pid_b={pid_b}, pid_a={pid_a}")
    # pdb.set_trace()
    # ------------------------------
    # Load input indices to gather a_key, a_value
    offs_input_coreset_indices_m = tl.arange(0, BLOCK_SIZE_M) % M
    input_indices_ptrs = input_indices_ptr + (
        pid_b * stride_input_indices_b
        + pid_a * stride_input_indices_a
        + offs_input_coreset_indices_m * stride_input_indices_m
    )
    input_indices = tl.load(input_indices_ptrs)

    # Load the shift tensor corresponding to the a-th halving problem in the b-th batch
    offs_shift_m = tl.arange(0, BLOCK_SIZE_M // 2)
    shift_ptrs = shift_ptr + (
        pid_b * stride_shift_b + pid_a * stride_shift_a + offs_shift_m * stride_shift_m
    )
    shift = tl.load(shift_ptrs, mask=offs_shift_m < (M // 2), other=0.0)
    # Call _exp_kernel_single_block to compute the Kdiff tensor
    # pdb.set_trace()
    Kdiff, new_shift = _exp_kernel_single_block_2(
        a_key_ptr + pid_b * stride_a_key_b,
        a_value_ptr + pid_b * stride_a_value_b,
        input_indices,
        shift,
        b_sqd_ptr + pid_b * stride_b_sqd_b,
        softmax_temp,
        E,
        D,
        INIT,
        stride_a_key_m,
        stride_a_key_e,
        stride_a_value_m,
        stride_a_value_d,
        BLOCK_SIZE_M,
        # NOTE: use BLOCK_SIZE_M = BLOCK_SIZE_N to ensure Kdiff is a square matrix
        BLOCK_SIZE_M,
        BLOCK_SIZE_E,
        BLOCK_SIZE_D,
    )
    # pdb.set_trace()
    # Write back Kdiff for the a-th halving problem in the b-th batch (of shape [M // 2, M // 2]) to HBM
    offs_Kdiff_m = tl.arange(0, BLOCK_SIZE_M // 2)
    offs_Kdiff_n = tl.arange(0, BLOCK_SIZE_M // 2)
    Kdiff_ptrs = (
        Kdiff_ptr
        + pid_b * stride_Kdiff_b
        + pid_a * stride_Kdiff_a
        + stride_Kdiff_m * offs_Kdiff_m[:, None]
        + stride_Kdiff_n * offs_Kdiff_n[None, :]
    )
    Kdiff_mask = (offs_Kdiff_m[:, None] < M // 2) & (offs_Kdiff_n[None, :] < M // 2)
    tl.store(Kdiff_ptrs, Kdiff, mask=Kdiff_mask)

    # ------------------------------
    # Initialize coreset_sum_diff for the a-th halving problem in the b-th batch
    offs_coreset_sum_diff_m = tl.arange(0, BLOCK_SIZE_M // 2)
    coreset_sum_diff_ptrs = coreset_sum_diff_ptr + (
        pid_b * stride_coreset_sum_diff_b
        + pid_a * stride_coreset_sum_diff_a
        + offs_coreset_sum_diff_m[None, :] * stride_coreset_sum_diff_m
    )
    coreset_sum_diff = tl.load(
        coreset_sum_diff_ptrs,
        mask=offs_coreset_sum_diff_m[None, :] < (M // 2),
        other=0.0,
    )
    if INIT:
        # Overwrite instead of `*= 0`: the buffer is torch.empty and may hold
        # NaN/Inf garbage (0 * NaN = NaN would poison the discrepancy state,
        # making halving outcomes depend on ambient allocator contents).
        coreset_sum_diff = tl.zeros_like(coreset_sum_diff)
    else:
        # Scale coreset_sum_diff by exp(shift - new_shift)
        coreset_sum_diff = coreset_sum_diff * tl.exp(
            tl.expand_dims(shift - new_shift, axis=0)
        )
    # Write back coreset_sum_diff for the a-th halving problem in the b-th batch
    tl.store(
        coreset_sum_diff_ptrs,
        coreset_sum_diff,
        mask=offs_coreset_sum_diff_m[None, :] < (M // 2),
    )
    # Write back new_shift for the a-th halving problem in the b-th batch
    tl.store(shift_ptrs, new_shift, mask=offs_shift_m < (M // 2))

    # Load rt_diag_cummax for the a-th halving problem in the b-th batch
    rt_diag_cummax_ptrs = rt_diag_cummax_ptr + (
        pid_b * stride_rt_diag_cummax_b + pid_a * stride_rt_diag_cummax_a
    )
    rt_diag_cummax = tl.load(rt_diag_cummax_ptrs)
    if INIT:
        # Overwrite instead of `*= 0`: torch.empty garbage may be NaN/Inf
        rt_diag_cummax = tl.zeros_like(rt_diag_cummax)
    # Call halve_K
    # pdb.set_trace()
    new_rt_diag_cummax = _halve_K(
        shift_ptr + pid_b * stride_shift_b + pid_a * stride_shift_a,
        Kdiff_ptr + pid_b * stride_Kdiff_b + pid_a * stride_Kdiff_a,
        uniform_ptr + pid_b * stride_uniform_b + pid_a * stride_uniform_a,
        coreset_sum_diff_ptr
        + pid_b * stride_coreset_sum_diff_b
        + pid_a * stride_coreset_sum_diff_a,
        rt_diag_cummax,
        swap_points_ptr + pid_b * stride_swap_points_b + pid_a * stride_swap_points_a,
        coreset_sum_diff,
        M,
        stride_shift_m,
        stride_Kdiff_m,
        stride_Kdiff_n,
        stride_uniform_m,
        stride_coreset_sum_diff_m,
        stride_swap_points_m,
        BLOCK_SIZE_M,
        INIT,
    )
    # Write back new_rt_diag_cummax for the a-th halving problem in the b-th batch
    # NOTE: new_rt_diag_cummax will be bogus if BLOCK_SIZE_M != M, but this only occurs for the last block,
    # after which we don't need it anymore
    tl.store(rt_diag_cummax_ptrs, new_rt_diag_cummax)

    # -----------------------------------------------------------
    # Epilogue: adjust coreset_indices so that they are valid indices for a_key, a_value
    # not input_indices, which may be a subset of a_key, a_value
    # pdb.set_trace()
    offs_swap_points_m = tl.arange(0, BLOCK_SIZE_M // 2) % (M // 2)
    swap_points_ptrs = swap_points_ptr + (
        pid_b * stride_swap_points_b
        + pid_a * stride_swap_points_a
        + offs_swap_points_m * stride_swap_points_m
    )
    swap_points = tl.load(swap_points_ptrs)
    if symmetrize_rv_ptr is not None:
        # compute xor between swap_points and symmetrize_rv
        symmetrize_rv = tl.load(symmetrize_rv_ptr)
        swap_points = swap_points ^ symmetrize_rv
    coreset_indices = input_indices.gather(2 * offs_swap_points_m + swap_points, 0)
    offs_coreset_indices_m = tl.arange(0, BLOCK_SIZE_M // 2)
    coreset_indices_ptrs = coreset_indices_ptr + (
        pid_b * stride_coreset_indices_b
        + pid_a * stride_coreset_indices_a
        + offs_coreset_indices_m * stride_coreset_indices_m
    )
    tl.store(
        coreset_indices_ptrs, coreset_indices, mask=offs_coreset_indices_m < (M // 2)
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n},
            num_stages=s,
            num_warps=w,
        )
        for m, n, s, w in [
            (128, 256, 3, 8),
            (64, 256, 4, 4),
            (128, 128, 4, 4),
            (128, 64, 4, 4),
            (64, 128, 4, 4),
            (128, 32, 4, 4),
            (64, 32, 5, 2),
            (32, 64, 5, 2),
            # Good configs for fp8 inputs.
            (256, 128, 3, 8),
            (256, 64, 4, 4),
        ]
    ],
    key=["M", "N", "E", "D"],
)
@triton.jit
def _batch_flash_psi(
    a_key_ptr,
    a_value_ptr,
    b_key_ptr,
    b_value_ptr,
    shift_ptr,
    b_sqd_ptr,
    softmax_temp: tl.constexpr,
    coreset_sum_diff_ptr,
    swap_points_ptr,
    col_input_indices_ptr,
    row_input_indices_ptr,
    # Matrix dimensions
    A,
    M,
    N,
    E: tl.constexpr,
    D: tl.constexpr,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_a_key_b,
    stride_a_key_m,
    stride_a_key_e,  #
    stride_b_key_b,
    stride_b_key_e,
    stride_b_key_n,  #
    stride_a_value_b,
    stride_a_value_m,
    stride_a_value_d,  #
    stride_b_value_b,
    stride_b_value_d,
    stride_b_value_n,  #
    stride_shift_b,
    stride_shift_a,
    stride_shift_m,
    stride_b_sqd_b,  #
    stride_coreset_sum_diff_b,
    stride_coreset_sum_diff_a,
    stride_coreset_sum_diff_m,
    stride_coreset_sum_diff_k,  #
    stride_swap_points_b,
    stride_swap_points_a,
    stride_swap_points_m,  #
    stride_col_input_indices_b,
    stride_col_input_indices_a,
    stride_col_input_indices_m,  #
    stride_row_input_indices_b,
    stride_row_input_indices_a,
    stride_row_input_indices_m,  #
    MIN_BLOCK_SIZE: tl.constexpr,
    INIT: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  #
):
    """Kernel for computing the partial psi vector for a single block

    Args:
        a_key_ptr: pointer to the key matrix A
        a_value_ptr: pointer to the value matrix A
        b_key_ptr: pointer to the key matrix B
        b_value_ptr: pointer to the value matrix B
        shift_ptr: pointer to the shift matrix
        b_sqd_ptr: pointer to the b_sqd matrix
        coreset_sum_diff_ptr: pointer to the coreset_sum_diff matrix
        swap_points_ptr: pointer to the swap_points matrix
        col_input_indices_ptr: pointer to the coreset_indices matrix
            that defines the column inputs
        row_input_indices_ptr: pointer to the coreset_indices matrix
            that defines the row inputs

    """
    pid_ba = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = pid_ba // A
    pid_a = pid_ba % A

    # pdb.set_trace()
    # ------------------------------
    # Compute psi for the m-th row block of the a-th halving problem in the b-th batch
    acc = tl.zeros((BLOCK_SIZE_M // 2, MIN_BLOCK_SIZE), dtype=tl.float32)
    # Load shift for the m-th row block of the a-th halving problem in the b-th batch
    # NOTE: if INIT is True, shift will be ignored (and initialized to 0)
    offs_shift_m = pid_m * (BLOCK_SIZE_M // 2) + tl.arange(0, BLOCK_SIZE_M // 2)
    shift_ptrs = shift_ptr + (
        pid_b * stride_shift_b + pid_a * stride_shift_a + offs_shift_m * stride_shift_m
    )
    shift_mask = offs_shift_m < (M // 2)
    m_i = tl.load(shift_ptrs, mask=shift_mask, other=0.0)
    if INIT:
        # Overwrite instead of `*= 0`: torch.empty garbage may be NaN/Inf
        m_i = tl.zeros_like(m_i)

    # Load the input_indices for the m-th row block of the a-th halving problem of the b-th batch
    offs_input_indices_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    row_input_indices_ptrs = row_input_indices_ptr + (
        pid_b * stride_row_input_indices_b
        + pid_a * stride_row_input_indices_a
        + offs_input_indices_m * stride_row_input_indices_m
    )
    row_input_indices = tl.load(row_input_indices_ptrs)
    # Use input_indices to index into the b-th batch of a_key
    offs_e = tl.arange(0, E)
    a_key_ptrs = a_key_ptr + (
        pid_b * stride_a_key_b
        + row_input_indices[:, None] * stride_a_key_m
        + offs_e[None, :] * stride_a_key_e
    )
    a_key = tl.load(a_key_ptrs)
    # Use input_indices to index into the b-th batch of a_value
    offs_d = tl.arange(0, D)
    a_value_ptrs = a_value_ptr + (
        pid_b * stride_a_value_b
        + row_input_indices[:, None] * stride_a_value_m
        + offs_d[None, :] * stride_a_value_d
    )
    a_value = tl.load(a_value_ptrs)
    # Compute pointers to the initial column block of the a-th halving problem of the b-th batch
    offs_input_indices_n = tl.arange(0, BLOCK_SIZE_N)
    # Compute pointers to the initial swap points for the a-th halving problem of the b-th batch
    # HACK: We can only call tl.dot on matrices with sides >= 16, so we broadcast to a block of size
    # [BLOCK_SIZE_N // 2, 16] and then mask the last 15 columns
    offs_swap_points_m = tl.arange(0, BLOCK_SIZE_N // 2)
    swap_points_ptrs = swap_points_ptr + (
        pid_b * stride_swap_points_b
        + pid_a * stride_swap_points_a
        + offs_swap_points_m[:, None] * stride_swap_points_m
    )
    # Iterate over column blocks of the a-th halving problem of the b-th batch
    for i in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # Load the indices for the i-th column block of the a-th halving problem of the b-th batch
        col_input_indices_ptrs = col_input_indices_ptr + (
            pid_b * stride_col_input_indices_b
            + pid_a * stride_col_input_indices_a
            + (offs_input_indices_n % N) * stride_col_input_indices_m
        )
        col_input_indices = tl.load(col_input_indices_ptrs)
        # Use col_input_indices to index into the i-th column block of the a-th halving problem
        # of the b-th batch of b_key and b_value
        b_key_ptrs = b_key_ptr + (
            pid_b * stride_b_key_b
            + offs_e[:, None] * stride_b_key_e
            + col_input_indices[None, :] * stride_b_key_n
        )
        b_value_ptrs = b_value_ptr + (
            pid_b * stride_b_value_b
            + offs_d[:, None] * stride_b_value_d
            + col_input_indices[None, :] * stride_b_value_n
        )
        b_key = tl.load(b_key_ptrs)
        b_value = tl.load(b_value_ptrs)
        # Call _exp_kernel_single_block to compute the Kdiff tensor
        Kdiff, m_ij = _exp_kernel_single_block(
            a_key,
            b_key,
            a_value,
            b_value,
            m_i,
            b_sqd_ptr + pid_b * stride_b_sqd_b,
            softmax_temp,
            INIT and (i == 0),
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
        # Compute eta = 1 - 2 * swap_points
        swap_points = tl.load(
            swap_points_ptrs, mask=offs_swap_points_m[:, None] < (N // 2), other=-1
        )
        eta = tl.where(
            swap_points >= 0,
            1 - 2 * swap_points,
            0,
        )
        # Rescale the accumulator by exp(m_i - m_ij)
        acc = acc * tl.exp((m_i - m_ij).expand_dims(1))
        # Update the output accumulator with Kdiff @ eta
        acc = tl.dot(
            Kdiff,
            eta.to(tl.float32).broadcast_to((BLOCK_SIZE_N // 2, MIN_BLOCK_SIZE)),
            acc,
        )
        # Update m_i for the next column block
        m_i = m_ij
        # Advance pointers
        b_key_ptrs += BLOCK_SIZE_N * stride_b_key_n
        b_value_ptrs += BLOCK_SIZE_N * stride_b_value_n
        swap_points_ptrs += (BLOCK_SIZE_N // 2) * stride_swap_points_m
        # col_input_indices_ptrs += BLOCK_SIZE_N * stride_col_input_indices_m
        offs_input_indices_n += BLOCK_SIZE_N

    # Write back the partial psi vector for the m-th row block of the a-th halving problem in the b-th batch
    offs_psi_m = pid_m * BLOCK_SIZE_M // 2 + tl.arange(0, BLOCK_SIZE_M // 2)
    offs_psi_k = tl.arange(0, MIN_BLOCK_SIZE)
    psi_ptrs = coreset_sum_diff_ptr + (
        pid_b * stride_coreset_sum_diff_b
        + pid_a * stride_coreset_sum_diff_a
        + stride_coreset_sum_diff_m * offs_psi_m[:, None]
        + stride_coreset_sum_diff_k * offs_psi_k[None, :]
    )
    tl.store(
        psi_ptrs, acc, mask=(offs_psi_m[:, None] < (M // 2)) & (offs_psi_k[None, :] < 1)
    )
    # Write back shift for the m-th row block of the a-th halving problem in the b-th batch
    tl.store(shift_ptrs, m_i, mask=shift_mask)


M_TO_M_INNER = {
    128: 64,
    256: 64,
    512: 64,
    1024: 128,
    2048: 128,
    4096: 128,
}


def block_halve(
    a_key: torch.Tensor,
    b_key: torch.Tensor,
    a_value: torch.Tensor,
    b_value: torch.Tensor,
    halve_prob: float,
    b_sqd: torch.Tensor,
    M_inner: int = 64,
    input_indices: torch.Tensor | None = None,  # New parameter
    softmax_temp: float = 1.0,
    symmetrize: bool = True,
    uniforms_buf: torch.Tensor = None,
    symmetrize_rv: torch.Tensor = None,
):
    """Blocked implementation of halve.

    Args:
        a_key: tensor of shape [B, n, E]
        b_key: tensor of shape [B, n, M]
        a_value: tensor of shape [B, n, D]
        b_value: tensor of shape [B, n, M]
        halve_prob: used to compute delta = halve_prob * M^2
        b_sqd: tensor of shape [B, 1, 1] accepted by exp kernel;
            typically represents the maximum squared value vector inf norm
        M_inner: int, the size of the halve_K block
        input_indices: tensor of shape [B, A, M], where A is the number of halving problems
            per batch and M is the size of each halving problem
        softmax_temp: scalar, softmax temperature for the exp kernel
        symmetrize: if True, randomly choose between coreset and its complement
        uniforms_buf: pre-allocated uniform random variables of shape [B, A, M // 2]
        symmetrize_rv: pre-allocated symmetrize random variable. If none, symmetrize is false.

    Returns:
        coreset: tensor of shape [B, A, M // 2] containing the output coreset's indices in a_key and a_value

    """
    # pdb.set_trace()
    E = a_key.shape[2]
    D = a_value.shape[2]
    B, A, M = input_indices.shape
    assert M % 2 == 0, f"M must be even but got M={M}"
    if uniforms_buf is None:
        num_points_in_coreset = M // 2
        log_multiplier = 0.5 + math.log(2 * M / halve_prob)
        uniforms = torch.empty(
            B, A, num_points_in_coreset, device=a_key.device
        ).uniform_(-log_multiplier, log_multiplier)
    else:
        # NOTE: uniforms_buf may be larger than needed; slice to correct size
        assert uniforms_buf.shape[0] == B
        assert uniforms_buf.shape[1] >= A
        assert uniforms_buf.shape[2] == M // 2
        uniforms = uniforms_buf[:, :A, : M // 2]

    if symmetrize:
        # Draw bernoulli variable if symmetrize_rv is None
        symmetrize_rv = (
            torch.empty(1, device=a_key.device, dtype=torch.bool).bernoulli_()
            if symmetrize_rv is None
            else symmetrize_rv
        )

    if M in M_TO_M_INNER:
        # Use one of the pre-tuned M_inner values
        M_inner = M_TO_M_INNER[M]
    else:
        M_inner = max(min(M_inner, M), MIN_BLOCK_SIZE)
    # Allocate Kdiff
    Kdiff = torch.empty(
        (B, A, M_inner // 2, M_inner // 2), device=a_key.device, dtype=torch.float32
    )
    # Allocate shift per row of Kdiff
    shift = torch.empty((B, A, M // 2), device=a_key.device, dtype=torch.float32)
    # Allocate coreset_sum_diff
    coreset_sum_diff = torch.empty(
        (B, A, M // 2, MIN_BLOCK_SIZE), device=a_key.device, dtype=torch.float32
    )
    # Allocate rt_diag_cummax and initialize to 0
    rt_diag_cummax = torch.empty((B, A), device=a_key.device, dtype=torch.float32)
    # Allocate coreset indices
    coreset_indices = torch.empty(
        (B, A, M // 2), device=a_key.device, dtype=torch.int32
    )
    # Allocate swap points
    swap_points = torch.empty((B, A, M // 2), device=a_key.device, dtype=torch.int32)

    # iterate over column blocks
    for lo in range(0, M, M_inner):
        # pdb.set_trace()
        hi = min(lo + M_inner, M)
        half_lo, half_hi = lo // 2, hi // 2
        # Select the on-band blocks
        shift_on_band = shift[:, :, half_lo:half_hi]
        coreset_sum_diff_on_band = coreset_sum_diff[:, :, half_lo:half_hi]
        uniforms_on_band = uniforms[:, :, half_lo:half_hi]
        coreset_indices_on_band = coreset_indices[:, :, half_lo:half_hi]
        input_indices_on_band = input_indices[:, :, lo:hi]
        swap_points_on_band = swap_points[:, :, half_lo:half_hi]

        # --- apply fused halve kernel for i-th diagonal block ---
        def grid(META):
            return (B * A,)

        _batch_halve_inner[grid](
            a_key,
            a_value,
            shift_on_band,
            b_sqd,
            softmax_temp,
            Kdiff,
            uniforms_on_band,
            coreset_sum_diff_on_band,
            rt_diag_cummax,
            coreset_indices_on_band,
            input_indices_on_band,  # New parameter
            swap_points_on_band,
            symmetrize_rv,
            A,
            hi - lo,  # M_inner,
            E,
            D,
            a_key.stride(0),
            a_key.stride(1),
            a_key.stride(2),  #
            a_value.stride(0),
            a_value.stride(1),
            a_value.stride(2),  #
            shift_on_band.stride(0),
            shift_on_band.stride(1),
            shift_on_band.stride(2),  #
            b_sqd.stride(0),  #
            Kdiff.stride(0),
            Kdiff.stride(1),
            Kdiff.stride(2),
            Kdiff.stride(3),  #
            uniforms_on_band.stride(0),
            uniforms_on_band.stride(1),
            uniforms_on_band.stride(2),  #
            coreset_sum_diff_on_band.stride(0),
            coreset_sum_diff_on_band.stride(1),
            coreset_sum_diff_on_band.stride(2),  #
            coreset_sum_diff_on_band.stride(3),  #
            rt_diag_cummax.stride(0),
            rt_diag_cummax.stride(1),  #
            coreset_indices_on_band.stride(0),
            coreset_indices_on_band.stride(1),
            coreset_indices_on_band.stride(2),  #
            input_indices_on_band.stride(0),  # input_indices strides
            input_indices_on_band.stride(1),
            input_indices_on_band.stride(2),  #
            swap_points_on_band.stride(0),
            swap_points_on_band.stride(1),
            swap_points_on_band.stride(2),  #
            lo == 0,  # INIT
            BLOCK_SIZE_M=M_inner,
        )
        # pdb.set_trace()
        if hi < M:
            # -- compute psi for the off-diagonal blocks in the i-th column ---
            # 2D launch kernel where each row block gets its own program.
            # Similar to FlashAttention, our implementation processes columns sequentially.
            def grid(META):
                return (B * A, triton.cdiv(M - hi, META["BLOCK_SIZE_M"]))

            # Select the off-band blocks
            input_indices_on_band = input_indices[:, :, lo:hi]
            input_indices_off_band = input_indices[:, :, hi:]
            shift_off_band = shift[:, :, half_hi:]
            coreset_sum_diff_off_band = coreset_sum_diff[:, :, half_hi:]
            _batch_flash_psi[grid](
                a_key,
                a_value,
                b_key,
                b_value,
                shift_off_band,
                b_sqd,
                softmax_temp,
                coreset_sum_diff_off_band,
                swap_points_on_band,
                input_indices_on_band,  # New parameter
                input_indices_off_band,
                A,
                M - hi,
                hi - lo,
                E,
                D,
                a_key.stride(0),
                a_key.stride(1),
                a_key.stride(2),
                b_key.stride(0),
                b_key.stride(1),
                b_key.stride(2),
                a_value.stride(0),
                a_value.stride(1),
                a_value.stride(2),
                b_value.stride(0),
                b_value.stride(1),
                b_value.stride(2),
                shift_off_band.stride(0),
                shift_off_band.stride(1),
                shift_off_band.stride(2),  #
                b_sqd.stride(0),
                coreset_sum_diff_off_band.stride(0),
                coreset_sum_diff_off_band.stride(1),
                coreset_sum_diff_off_band.stride(2),
                coreset_sum_diff_off_band.stride(3),  #
                swap_points_on_band.stride(0),
                swap_points_on_band.stride(1),
                swap_points_on_band.stride(2),  #
                input_indices_on_band.stride(0),  # input_indices strides
                input_indices_on_band.stride(1),
                input_indices_on_band.stride(2),  #
                input_indices_off_band.stride(0),  # input_indices strides
                input_indices_off_band.stride(1),
                input_indices_off_band.stride(2),  #
                MIN_BLOCK_SIZE,
                lo == 0,  # INIT
            )

    return None, coreset_sum_diff, coreset_indices
