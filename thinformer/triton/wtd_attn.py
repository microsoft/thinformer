"""Triton implementation of Flash Attention 2 (without TMA support).

Reference:
https://github.com/triton-lang/triton/blob/8bd8035559d4d2fa387149eecdf10d545d8a3d03/python/tutorials/06-fused-attention.py
"""

import torch
import triton
import triton.language as tl


def is_hip():
    """Return True if the active Triton backend is HIP (AMD)."""
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    """Return True if the active Triton backend is CUDA (NVIDIA)."""
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    lo,
    hi,
    qk_scale,  #
    BLOCK_N: tl.constexpr,  #
    ON_BAND: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
):
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        # -- compute qk ----
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        qk = tl.dot(q, k)
        if ON_BAND:
            mask = (lo + offs_m)[:, None] >= (start_n + offs_n[None, :])
        else:
            mask = (start_n + offs_n[None, :]) < hi
        qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(p, v.to(tl.float32), acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_idx_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_ptr,
    V_ptr,  #
    level_idx_ptr,  #
    qk_scale,  #
    log2_weight,  #
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    offs_n: tl.constexpr,  #
    N_CTX,
    stride_kn,
    stride_kk,
    stride_vk,
    stride_vn,
    stride_level_idx_n,
):
    # block index pointers
    # NOTE: prevent indexing beyond N_CTX. The index buffer holds exactly N_CTX
    # entries per (batch, head); mask every load so out-of-range lanes are not
    # dereferenced (N_CTX need not be a multiple of BLOCK_N).
    idx_ptrs = level_idx_ptr + offs_n * stride_level_idx_n
    # offsets
    offs_dim = tl.arange(0, HEAD_DIM)
    # loop over k, v and update accumulator
    for start_n in tl.range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask = (start_n + offs_n) < N_CTX
        # -- compute k, v pointers
        indices = tl.load(idx_ptrs, mask=mask, other=0)
        # K_ptrs has shape (HEAD_DIM, BLOCK_N)
        K_ptrs = K_ptr + (offs_dim[:, None] * stride_kk + indices[None, :] * stride_kn)
        # V_ptrs has shape (BLOCK_N, HEAD_DIM)
        V_ptrs = V_ptr + (indices[:, None] * stride_vk + offs_dim[None, :] * stride_vn)
        # -- compute qk ----
        k = tl.load(K_ptrs)
        qk = tl.dot(q, k)
        qk = qk * qk_scale + tl.where(mask[None, :], 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk + log2_weight)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_ptrs)
        acc = tl.dot(p, v.to(tl.float32), acc)
        # update m_i and l_i
        m_i = m_ij
        idx_ptrs += BLOCK_N * stride_level_idx_n
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
if True:
    configs = [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
        for BM in [16, 32, 64, 128]
        for BN in [16, 32, 64]
        for s in [3, 4, 7]
        for w in [4, 8]
    ]
else:
    configs = [triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_stages=3, num_warps=4)]


def _prune_undersampling_block_m(configs, named_args, **kwargs):
    """Undersampling: the query tile must not exceed the post-subsampling bucket
    ``bucket_size >> log2_subsampling_factor``.  The off-band level-(-1) term
    (``n_off``) changes every such sub-bucket, so a program spanning more than
    one would collapse it (dropping the subsampled off-band for every sub-bucket
    after the first).  Keep only configs whose ``BLOCK_M`` is at least the
    sub-bucket, so the effective tile ``min(BLOCK_M, sub-bucket)`` equals it.
    Oversampling (``log2_subsampling_factor == 0``) is unaffected.
    """

    def _get(name):
        if name in named_args:
            return named_args[name]
        return kwargs[name]

    log2_ss = _get("log2_subsampling_factor")
    if log2_ss and log2_ss > 0:
        bucket_sub = _get("bucket_size") >> log2_ss
        kept = [c for c in configs if c.kwargs.get("BLOCK_M", 0) >= bucket_sub]
        if kept:
            return kept
    return configs


@triton.autotune(
    configs,
    key=["bucket_size", "m", "coreset_size", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_undersampling_block_m},
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    level_idx,
    input_indices_round,
    bucket_size,
    m,
    coreset_size,
    log2_subsampling_factor: tl.constexpr,
    l,
    r,
    n_first,
    n_last,
    n_warm,
    Out,  #
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    stride_level_idx_z,
    stride_level_idx_h,
    stride_level_idx_n,  #
    stride_input_indices_round_z,
    stride_input_indices_round_h,
    stride_input_indices_round_n,  #
    H,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_KEY_VALUE_GROUPS: tl.constexpr,  #
    UNDERSAMPLING: tl.constexpr,  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    # qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Effective query tile: in the undersampling regime it must not exceed the
    # post-subsampling bucket (bucket_size >> log2_subsampling_factor) so every
    # query in a program shares one off-band sub-bucket (see
    # _prune_undersampling_block_m).  Oversampling: log2_ss == 0 -> unchanged.
    q_block = min(BLOCK_M, bucket_size >> log2_subsampling_factor)
    # block pointers
    # NOTE: https://github.com/triton-lang/triton/issues/3890 suggests that shape and order do not matter
    # and only affect the compiler's performance.
    kv_shape_bound = n_first + min(l + (start_m + 1) * q_block, r) + n_last
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + off_z * stride_vz + (off_h // NUM_KEY_VALUE_GROUPS) * stride_vh,
        shape=(kv_shape_bound, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + off_z * stride_kz + (off_h // NUM_KEY_VALUE_GROUPS) * stride_kh,
        shape=(HEAD_DIM, kv_shape_bound),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    # initialize offsets
    # TODO: add wrap around
    offs_m = tl.arange(0, BLOCK_M)
    # TODO: add wrap around
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    offs_d = tl.arange(0, HEAD_DIM)
    q_m = start_m * q_block + offs_m
    mask_q = q_m < (r - l)
    Q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h * stride_qh
        + q_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q = tl.load(Q_ptrs, mask=mask_q[:, None], other=0.0)
    # NOTE: compute attention for level-0 coreset (or level -1 coreset in undersampling regime)
    # stage 1: off-band
    # -- Attention sink --
    if n_first > 0:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            0,
            n_first,
            qk_scale,  #
            BLOCK_N,  #
            False,
            offs_m,
            offs_n,
        )
    if UNDERSAMPLING:
        # In the undersampling regime, restrict the off-band level-(-1) attention to
        # the subsampled positions in input_indices_round, weighted with
        # log2_subsampling_factor (= log2_subsampling_factor + 1 + level evaluated at
        # level = -1). The on-band diagonal block stays raw / exact below.
        # input_indices_round is indexed from 0 within the round, so use
        # round-relative offsets (no `l +` prefix). stride = 2 ** log2_subsampling_factor.
        t_rel_off = start_m * q_block
        ell_rel_off = t_rel_off // bucket_size * bucket_size
        # The n_last sliding-window keys immediately before this block
        # (kv indices [block_start, block_start + n_last)) are raw. The
        # non-undersampling off-band covers them via its `+ n_last` upper
        # bound; without this pass every query in an undersampled round
        # misses its most recent raw keys.
        if n_last > 0:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,  #
                n_first + l + t_rel_off,
                n_first + l + t_rel_off + n_last,
                qk_scale,  #
                BLOCK_N,  #
                False,
                offs_m,
                offs_n,
            )
        # Both ell_rel_off and t_rel_off are multiples of 2 ** log2_subsampling_factor,
        # so right-shifting yields exact integer subsampled-index bounds.
        # n_warm > 0 only when mstar == 0, where the whole round fits in one
        # bucket (ell_rel_off == 0) and E is prepended to input_indices_round.
        n_off = n_warm + ((t_rel_off - ell_rel_off) >> log2_subsampling_factor)
        if n_off > 0:
            K_ptr_off = (
                K + off_z * stride_kz + (off_h // NUM_KEY_VALUE_GROUPS) * stride_kh
            )
            V_ptr_off = (
                V + off_z * stride_vz + (off_h // NUM_KEY_VALUE_GROUPS) * stride_vh
            )
            input_indices_round_ptr = (
                input_indices_round
                + off_z * stride_input_indices_round_z
                + (off_h // NUM_KEY_VALUE_GROUPS) * stride_input_indices_round_h
                + (ell_rel_off >> log2_subsampling_factor)
                * stride_input_indices_round_n
            )
            acc, l_i, m_i = _attn_idx_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_ptr_off,
                V_ptr_off,  #
                input_indices_round_ptr,  #
                qk_scale,  #
                tl.cast(log2_subsampling_factor, tl.float32),  #
                HEAD_DIM,
                BLOCK_N,  #
                offs_n,  #
                n_off,
                stride_kn,
                stride_kk,
                stride_vk,
                stride_vn,
                stride_input_indices_round_n,
            )
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            n_first
            + (l + start_m * q_block)
            // bucket_size
            * bucket_size,  # start of level-0 coreset
            n_first + l + start_m * q_block + n_last,
            qk_scale,  #
            BLOCK_N,  #
            False,
            offs_m,
            offs_n,
        )
    # stage 2: on-band
    # barrier makes it easier for compielr to schedule the
    # two loops independently
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,  #
        n_first + l + start_m * q_block + n_last,
        n_first + l + start_m * q_block + BLOCK_M + n_last,
        qk_scale,  #
        BLOCK_N,  #
        True,
        offs_m,
        offs_n,
    )

    # Attend over level coresets
    V_ptr = V + off_z * stride_vz + (off_h // NUM_KEY_VALUE_GROUPS) * stride_vh
    K_ptr = K + off_z * stride_kz + (off_h // NUM_KEY_VALUE_GROUPS) * stride_kh
    level_idx_ptr = (
        level_idx
        + off_z * stride_level_idx_z
        + (off_h // NUM_KEY_VALUE_GROUPS) * stride_level_idx_h
    )
    # adjust t according to the compression level g
    # in other words, we are computing the base-4 representation of t
    # and ignoring the trailing g + 1 digits
    t = start_m * q_block // bucket_size
    # -- Main loop over non-zero level coresets --
    level_length = tl.cast(
        (r - l) / tl.exp2(tl.cast(log2_subsampling_factor, tl.float32)), tl.int32
    )
    for level in tl.range(m):
        if level == m - 1:
            # last level: E coreset plus completed sibling outputs
            N_CTX = tl.cast(((t + 1) % 4) * coreset_size, tl.int32)
        else:
            N_CTX = tl.cast((t % 4) * coreset_size, tl.int32)
        # NOTE: if we pass N_CTX directly to _attn_idx_fwd_inner, the compiler will treat N_CTX = 0 always
        if N_CTX > 0:
            acc, l_i, m_i = _attn_idx_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_ptr,
                V_ptr,  #
                level_idx_ptr
                + tl.cast(
                    (t // 4) * 4 * coreset_size * stride_level_idx_n, tl.int32
                ),  #
                qk_scale,  #
                log2_subsampling_factor + 1 + tl.cast(level, tl.float32),  #
                HEAD_DIM,
                BLOCK_N,  #
                offs_n,  #
                N_CTX,
                stride_kn,
                stride_kk,
                stride_vk,
                stride_vn,
                stride_level_idx_n,
            )
        t //= 4
        coreset_size *= 2
        # Update idx to point to the next level indices,
        # which is half the size of the previous level indices minus any bucket_size remainder,
        # except the level `max_levels-1`, which adds the warm indices.
        # Since it's the last one, we don't care what the next level_idx_ptr is.
        two_to_level = tl.cast(tl.exp2(tl.cast(level, tl.float32)), tl.int32)
        # Bucket size at this level in the same units as level_length. In the
        # undersampling regime level_length counts subsampled positions while
        # bucket_size is in raw units (larger by 2^log2_subsampling_factor),
        # so shift it down; otherwise partial rounds floor to the wrong multiple
        # and every subsequent level segment is read at the wrong offset.
        level_bucket = (bucket_size >> log2_subsampling_factor) * two_to_level
        level_length = level_length // level_bucket * level_bucket
        level_length = level_length // 2
        level_idx_ptr += level_length * stride_level_idx_n

    # epilogue
    acc = acc / l_i[:, None]
    offs_d = tl.arange(0, HEAD_DIM)
    out_m = start_m * q_block + offs_m
    # Only this program's own stride of rows is valid (see flash_attn_levels):
    # rows beyond q_block exceed this program's K/V bound.
    mask_m = (offs_m < q_block) & (out_m < (r - l))
    Out_ptrs = (
        Out
        + off_z * stride_oz
        + off_h * stride_oh
        + out_m[:, None] * stride_om
        + offs_d[None, :] * stride_on
    )
    tl.store(Out_ptrs, acc.to(Out.type.element_ty), mask=mask_m[:, None])


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    level_idx: torch.Tensor,
    bucket_size: int,
    l: int,
    r: int,
    n_first: int,
    n_last: int,
    o: torch.Tensor,
    m: int,
    log2_subsampling_factor: int = 0,
    n_warm: int = 0,
    input_indices_round: torch.Tensor = None,
    num_key_value_groups: int = 1,
):
    """Triton implementation of FlashAttention2 over coreset levels

    NOTE: Assumes BLOCK_M <= 4 ** (g + 1) * inflation_factor. That is, each block of queries shares the same
    coreset levels (except for level 0, which uses normal causal attention).

    Args:
        q: query tensor of shape (B, H, M_CTX, HEAD_DIM)
        k: key tensor of shape (B, H, N_CTX, HEAD_DIM)
        v: value tensor of shape (B, H, N_CTX, HEAD_DIM)
        sm_scale: float
        level_idx: tensor of shape (B, H, coreset_size), representing the coreset indices
        bucket_size: initial bucket size
        l: int, start index for query tokens after sink
        r: int, end index for query tokens after sink
        n_first: int, size of attention sink
        n_last: int, size of sliding window
        o: output tensor of shape (B, H, M_CTX, HEAD_DIM)
        m: int, number of halving rounds (= capped compress depth m ^ mbar)
        log2_subsampling_factor: int, log2 of the stratified-subsampling step
            (0 outside the undersampling regime). bucket_size is in RAW
            positions; the subsampled compress bucket is
            bucket_size >> log2_subsampling_factor.
        n_warm: int, number of warm-coreset (E) points prepended to
            input_indices_round. Nonzero only when m == 0 (mbar == 0): E's
            points then carry multiplicity 2^log2_subsampling_factor, the same
            weight as a subsampled point, so they are attended through the
            off-band level-(-1) pass.
        input_indices_round: tensor of shape (B, H, n_subsampled) holding the
            raw positions kept by stratified subsampling for the current round.
            Required when log2_subsampling_factor > 0; pass None otherwise.
        num_key_value_groups: int, number of query heads that attend over each KV head
    Returns:
        None, writes output to `o`.

    """
    # context constraints
    assert k.shape[2] == v.shape[2]
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    extra_kern_args = {}
    # Tuning for AMD target
    if is_hip():
        waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

    # NOTE: BLOCK_M <= bucket_size is required to ensure that each block of queries
    # attends over the same coreset levels (except for the level 0, which uses normal causal attention)
    coreset_size = bucket_size // 2**log2_subsampling_factor // 2

    undersampling = log2_subsampling_factor > 0
    if undersampling:
        assert input_indices_round is not None, (
            "input_indices_round is required when log2_subsampling_factor > 0 "
            "(undersampling regime)."
        )
        assert input_indices_round.dim() == 3, (
            f"input_indices_round must be 3-D (B, H, n_subsampled); got {tuple(input_indices_round.shape)}."
        )
        input_indices_round_arg = input_indices_round
        stride_iir = input_indices_round.stride()
    else:
        # Pass the level_idx tensor as a placeholder (kernel won't read it under
        # `if UNDERSAMPLING:`); strides default to 0.
        input_indices_round_arg = level_idx
        stride_iir = (0, 0, 0)

    def grid(args):
        return (
            triton.cdiv(
                r - l, min(args["BLOCK_M"], bucket_size >> log2_subsampling_factor)
            ),
            q.shape[0] * q.shape[1],
            1,
        )

    _attn_fwd[grid](
        q,
        k,
        v,
        sm_scale,
        level_idx,
        input_indices_round_arg,
        bucket_size,
        m,
        coreset_size,
        log2_subsampling_factor,
        l,
        r,
        n_first,
        n_last,
        n_warm,
        o,  #
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),  #
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),  #
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),  #
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),  #
        level_idx.stride(0),
        level_idx.stride(1),
        level_idx.stride(2),  #
        stride_iir[0],
        stride_iir[1],
        stride_iir[2],  #
        q.shape[1],  #
        HEAD_DIM=HEAD_DIM_K,  #
        NUM_KEY_VALUE_GROUPS=num_key_value_groups,
        UNDERSAMPLING=undersampling,
        **extra_kern_args,
    )
