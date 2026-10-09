"""Implementation of Express that sequentially runs KH-Compress over the rounds."""

# standard imports
import torch
from torch import (
    inf,
)
from torch.linalg import vector_norm

# helper imports
from ..timing_utils import multi_events
from ..generic_utils import delta_compress, delta_warm

from .block_halve import block_halve
from .wtd_attn import attention


def khcompress(
    key: torch.Tensor,
    value: torch.Tensor,
    input_indices: torch.Tensor,
    b_sqd: torch.Tensor,
    bucket_size: int,
    mstar: int,
    delta: float = 0.5,
    m: int = None,
    alias: str = "khcompress",
    softmax_temp: float = 1.0,
    uniforms: torch.Tensor = None,
) -> torch.Tensor:
    """KH-Compress(g) with warm start

    KH-Compress implementation that takes in 3 x power-of-4 number of points and
    a warm-start coreset for the final halving round.

    Args:
        key: Tensor of shape [B * H, S, E]
        value: Tensor of shape [B * H, S, D]
        input_indices: Tensor of shape [B * H, n]
        b_sqd: Tensor of shape [B * H, 1, 1], max squared value vector inf norm
        bucket_size: Initial bucket size, int >= 0
        mstar: number of halving rounds (= capped compress depth m ^ mbar), int >= 0
        delta: overall Express failure-probability budget, scalar in [0,1]
        m: true (uncapped) even Express round level used for the delta
            schedule: algo:express runs Compress2(m ^ mbar, ..., halve_m), so
            the halving row halve_{m,i} is indexed by the true round level
            while only the depth is capped. Defaults to m (no cap).
        alias: Alias for the event name
        softmax_temp: scalar, softmax temperature for the exp kernel
        uniforms: dict of uniform RV buffers for each halving round

    Returns:
        levels: list of (coreset_indices, level) tuples, where each coreset_indices is
            a tensor of shape [B * H, coreset_size]

    """
    B_H, S = input_indices.shape
    if m is None:
        m = mstar
    # --- Main loop ---
    coreset_indices = None
    # initialize coreset
    levels = []
    for i in range(mstar):
        # Shape the key and value tensors into evenly sized buckets, so that we can call halve in a parallel fashion
        n = input_indices.shape[1]
        remainder_start = bucket_size * (n // bucket_size)
        remainder_end = n
        if remainder_start < remainder_end:
            input_indices = input_indices[:, :remainder_start]
        # Record the event for the i-th halving round with key.shape[1] buckets of size bucket_size
        input_indices = input_indices.view(B_H, -1, bucket_size)
        event_name = f"{alias}__halve-l{i}"
        with multi_events(event_name):
            _, _, coreset_indices = block_halve(
                key,
                key.mT,
                value,
                value.mT,
                delta_compress(delta, m, i, mstar),
                b_sqd,
                input_indices=input_indices,
                softmax_temp=softmax_temp,
                uniforms_buf=uniforms[event_name] if uniforms is not None else None,
            )
        # add coreset indices to coreset
        levels.append((coreset_indices.view(B_H, -1), i))
        bucket_size *= 2
        input_indices = coreset_indices.view(B_H, -1)
    # --- Epilogue ---
    return levels


def thinformer_express(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    nout: int,
    max_halving_rounds: int,
    softmax_temp: float,
    delta: float,
    n_first: int = 0,
    n_last: int = 0,
    num_key_value_groups: int = 1,
    alias: str = "express",
) -> torch.Tensor:
    """Forward pass of the ExpressAttention module.

    Args:
        query (torch.Tensor): Query tensor of shape (B, T, H, E).
        key (torch.Tensor): Key tensor of shape (B, S, H, E).
        value (torch.Tensor): Value tensor of shape (B, S, H, D).
        nout (int): Target cache size
        max_halving_rounds (int): Maximum number of halving rounds in each Express round
        softmax_temp (float): Softmax temperature for the attention computation.
        delta (float): Overall Express failure-probability budget.
        n_first (int): Number of attention sinks.
        n_last (int): Size of the sliding window.
        num_key_value_groups (int): Number of key-value groups.
        alias (str): Alias for the event name.

    Returns:
        torch.Tensor: Output tensor after applying Express attention.

    """
    # Uncapped depth degenerates once round m exceeds log2(4 * nout):
    # bucket_sub = (4 * nout) >> mstar reaches 0 and the bucket reshaping
    # in khcompress/block_halve fails ("M must be even").
    assert max_halving_rounds is not None, (
        "max_halving_rounds must be an int (the mbar cap); uncapped depth "
        "(None) is unsupported."
    )
    with multi_events(f"{alias}__prologue1"):
        # Allocate placeholder tensor for the output
        B, T, H, E = query.shape
        _, S, _, D = value.shape
        n_first_last = n_first + n_last
        num_key_value_heads = H // num_key_value_groups
        # Compute exact attention for the exact regime. Round m=0 of
        # algo:express stores raw points, so tokens through 4 * nout see
        # the full raw prefix and can use exact attention directly.
        exact_regime = n_first_last + 4 * nout
        # Allocate output tensor
        out = torch.empty(B, T, H, D, device=query.device, dtype=query.dtype)
        uniforms = None
    # -- Stage 1: Exact Regime --
    with multi_events(f"{alias}__attn0"):
        # We set is_causal=True since the attention matrix is square
        out[:, :exact_regime] = torch.nn.functional.scaled_dot_product_attention(
            query=query[:, :exact_regime].transpose(1, 2),
            key=key[:, :exact_regime].transpose(1, 2),
            value=value[:, :exact_regime].transpose(1, 2),
            dropout_p=0.0,
            scale=softmax_temp,
            is_causal=True,
            enable_gqa=(num_key_value_groups > 1),
        ).transpose(1, 2)
    # Return early if sequence length is in the exact regime
    if S <= exact_regime:
        return out

    with multi_events(f"{alias}__prologue2"):
        # Reshape key and value
        k_collapsed = key.transpose(1, 2).view(B * num_key_value_heads, S, E)
        v_collapsed = value.transpose(1, 2).view(B * num_key_value_heads, S, D)
        # Max squared value vector inf norm
        b_sqd = (
            vector_norm(
                v_collapsed[:, n_first : S - n_last],
                dim=(1, 2),
                ord=inf,
                keepdim=True,
            )
            ** 2
        )
        # initialize input indices
        undersampling_threshold = S
        input_indices = torch.arange(
            0,
            min(n_first_last + undersampling_threshold, S),
            device=key.device,
            dtype=torch.int32,
        )[None, :].expand(
            B * num_key_value_heads, min(n_first_last + undersampling_threshold, S)
        )

    # --- Stage 2: Main Loop ---
    # Round m (even) covers query positions (nout * 2^m, nout * 2^(m+2)]
    # after the sink: three siblings of nout * 2^m points each (algo:express).
    m = 2
    # Raw output of round 0 (the exact regime); halved twice at the start
    # of round 2 to form the initial Express coreset E.
    coreset_indices = input_indices[:, n_first : n_first + 4 * nout]
    mbar = max_halving_rounds
    while n_first + nout * 2**m + n_last < S:
        l = nout * 2**m
        r = min(nout * 2 ** (m + 2), S - n_first_last)
        # Capped compress depth (algo:express: Compress2(m ^ mbar, ...))
        mstar = min(m, mbar)
        # Stratified-subsampling step and its log2 (1 / 0 when uncapped)
        step = 2 ** (m - mstar)
        log2_subsampling_factor = m - mstar
        # Level-0 batch size of Compress2(mstar), in subsampled positions
        bucket_sub = (4 * nout) >> mstar
        # -- Halve phase for the completed round m-2: E and its three
        # sibling outputs (4 * nout points, all representing 2^(m-2) raw
        # points each) are halved twice down to nout
        with multi_events(f"{alias}__halve-m{m}"):
            cset_indices = coreset_indices.view(B * num_key_value_heads, 1, -1)
            halve_prob = delta_warm(delta, m - 2)
            for suffix in ["a", "b"]:
                _, _, cset_indices = block_halve(
                    k_collapsed,
                    k_collapsed.mT,
                    v_collapsed,
                    v_collapsed.mT,
                    halve_prob=halve_prob,
                    b_sqd=b_sqd,
                    M_inner=128,
                    input_indices=cset_indices.view(B * num_key_value_heads, 1, -1),
                    softmax_temp=softmax_temp,
                    uniforms_buf=uniforms[f"{alias}_express__halve-m{m}{suffix}"]
                    if uniforms is not None
                    else None,
                )
            cset_indices = cset_indices.view(B * num_key_value_heads, -1)
            assert cset_indices.shape[1] == nout, (
                f"E must have {nout} points after the halve phase, "
                f"got {cset_indices.shape[1]}."
            )
        # Call compress on points l:r
        with multi_events(f"{alias}__khcompress{m}"):
            if mstar == m:
                input_indices_round = input_indices[:, n_first + l : n_first + r]
            else:
                # Subsample(m, mbar): keep one uniform index per group of
                # `step` raw positions (stratified sampling; step is a
                # power of 4 so all coreset weights stay aligned).
                # FLOOR the strata count: on the final, sequence-truncated
                # round r is clamped to S - n_first_last, so (r - l) need
                # not be a multiple of step. The kernel's level_length and
                # khcompress both use floor((r - l) / step); a ceil'd extra
                # stratum here would shift every level-segment boundary.
                n_sub = (r - l) // step
                input_indices_round = torch.arange(
                    n_first + l,
                    n_first + l + n_sub * step,
                    step=step,
                    device=key.device,
                    dtype=torch.int32,
                )[None, :].expand(B * num_key_value_heads, n_sub) + torch.randint(
                    0,
                    step,
                    size=(B * num_key_value_heads, n_sub),
                    device=key.device,
                    dtype=torch.int32,
                )
            if mstar == 0:
                # mbar == 0: no compress rounds, so there are no level
                # coresets and the kernel's level loop is skipped (m=0).
                # E's nout points each carry multiplicity 2^m -- the same
                # weight as one stratified-subsampled point -- and the whole
                # round fits inside one bucket (span 3*nout*2^m <
                # bucket_size 4*nout*2^m, so ell_rel_off == 0). E is
                # therefore attended through the off-band level-(-1) pass by
                # prepending it to input_indices_round (n_warm below).
                levels = None
                input_indices_round = torch.cat(
                    (cset_indices, input_indices_round), dim=1
                )
            else:
                levels = khcompress(
                    k_collapsed,
                    v_collapsed,
                    input_indices_round,
                    b_sqd,
                    bucket_sub,
                    mstar,
                    delta=delta,
                    m=m,
                    alias=f"{alias}_express__compress-m{m}",
                    softmax_temp=softmax_temp,
                    uniforms=uniforms,
                )
        # Construct indices that we will attend to: levels 0..m-2 in order,
        # then [E ++ level-(m-1) sibling outputs] as the E-accumulating
        # last segment read by the kernel's ((t + 1) % 4) * coreset_size
        with multi_events(f"{alias}__gather_indices{m}"):
            if mstar == 0:
                # level_idx placeholder; kernel reads nothing when m == 0
                indices = cset_indices
            else:
                level_indices = [indices for indices, _ in levels[:-1]]
                level_indices.append(cset_indices)
                level_indices.append(levels[-1][0])
                indices = torch.cat(level_indices, dim=1)
        # -- Attend --
        with multi_events(f"{alias}__attn{m}"):
            # In the undersampling regime, the kernel restricts the off-band
            # level-(-1) attention to the subsampled raw positions in
            # `input_indices_round` (3-D view: B, H_kv, n_subsampled).
            if mstar < m:
                input_indices_round_arg = input_indices_round.view(
                    B, num_key_value_heads, -1
                )
            else:
                input_indices_round_arg = None
            attention(
                q=query[:, n_first_last + l : n_first_last + r].transpose(1, 2),
                k=key.transpose(1, 2),
                v=value.transpose(1, 2),
                sm_scale=softmax_temp,
                level_idx=indices.view(B, num_key_value_heads, -1),
                # Raw bucket size: subsampled compress bucket x step
                bucket_size=bucket_sub * step,
                l=l,
                r=r,
                n_first=n_first,
                n_last=n_last,
                o=out[:, n_first_last + l : n_first_last + r].transpose(1, 2),
                m=mstar,
                log2_subsampling_factor=log2_subsampling_factor,
                n_warm=nout if mstar == 0 else 0,
                input_indices_round=input_indices_round_arg,
                num_key_value_groups=num_key_value_groups,
            )
        # E for the next round: current E plus the three sibling outputs
        # (each of size nout at level m), to be halved twice next round.
        # mstar == 0: the sibling "outputs" are the subsampled points
        # themselves, already concatenated behind E above.
        if mstar == 0:
            coreset_indices = input_indices_round
        else:
            coreset_indices = torch.cat((cset_indices, levels[-1][0]), dim=1)
        m += 2

    return out
