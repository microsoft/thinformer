"""Triton implementation of KH-Compress."""

# standard imports
import torch
from torch import (
    arange,
    inf,
)
from torch.linalg import vector_norm
import math

# kernel halving imports
from .block_halve import block_halve

# helper functions
from ..timing_utils import multi_events
from ..memory_utils import profiler


# @torch.compile(mode="reduce-overhead", fullgraph=True)
def _khcompress(
    key: torch.Tensor,
    value: torch.Tensor,
    bucket_size: int,
    num_halving_rounds: int,
    halve_prob: float,
    symmetrize: bool,
    alias: str,
    M_inner: int = 64,
    input_indices: torch.Tensor = None,
    softmax_temp: float = 1.0,
    use_vbar_kbar: bool = False,
    uniforms: dict[int, torch.Tensor] = None,
    symmetrize_rvs: dict[int, torch.Tensor] = None,
) -> torch.Tensor:
    """KH-Compress

    Produces a coreset of size bucket_size * 4^(num_halving_rounds - 1) / 2.

    Args:
        key: Tensor of shape [B, S, E]
        value: Tensor of shape [B, S, D]
        bucket_size: initial bucket size for Compress
        num_halving_rounds: number of Compress halving rounds
        halve_prob: parameters for halve()
        symmetrize: if True, randomly choose between coreset and its complement
        alias: alias for the event
        M_inner: inner dimension for block halve
            If M_inner = 0, then non-blocking version of halve is used.
            If M_inner > 0, then blocking version of halve is used with block size
            max(M_inner, bucket_size // 2). That is, we always use 2x2 blocking, unless
            bucket_size is smaller than M_inner, in which case we use a single block
            of size bucket_size.
        input_indices: tensor of shape [B, S] containing the indices of the input points
        softmax_temp: scalar, softmax temperature for the exp kernel
        use_vbar_kbar: if True, use vbar and kbar to center the keys and values
        uniforms: pre-allocated uniform random variables
        symmetrize_rvs: pre-allocated symmetrize random variables

    Returns:
        key coreset tensor of shape [B, S//2^m, E] and
        value coreset tensor of shape [B, S//2^m, D]

    """
    # Max squared value vector inf norm
    b_sqd = vector_norm(value, dim=(1, 2), ord=inf, keepdim=True) ** 2

    B, S, E = key.shape
    _, _, D = value.shape

    # Execute Compress in bottom-up fashion by iteratively dividing input
    # into consecutive buckets of size bucket_size and halving each bucket.
    # Note that bucket_size grows by a factor of 2 on each halving round.
    for i in range(num_halving_rounds):
        input_indices = input_indices.view(B, -1, bucket_size)
        event_name = f"{alias}__halve{i}_{input_indices.shape[1]}_{bucket_size}"
        with profiler(event_name):
            with multi_events(event_name):
                # Only perform gather operations on the last halving round
                is_last_halve = i == num_halving_rounds - 1
                _, _, coreset_indices = block_halve(
                    key,
                    key.mT,
                    value,
                    value.mT,
                    halve_prob,
                    b_sqd,
                    M_inner=M_inner,
                    input_indices=input_indices,
                    softmax_temp=softmax_temp,
                    symmetrize=symmetrize,
                    uniforms_buf=uniforms[i],
                    symmetrize_rv=symmetrize_rvs[i],
                )
        bucket_size *= 2
        # Update input_indices for next halve call
        input_indices = coreset_indices

        if is_last_halve:
            with multi_events(f"{alias}__gather"):
                key = key.gather(
                    dim=1,
                    index=coreset_indices.view(B, -1)
                    .unsqueeze(-1)
                    .expand(-1, -1, E)
                    .to(torch.int64),
                )
                value = value.gather(
                    dim=1,
                    index=coreset_indices.view(B, -1)
                    .unsqueeze(-1)
                    .expand(-1, -1, D)
                    .to(torch.int64),
                )

    # events[f"{alias}_body"]["end"].record()
    return key, value


def khcompress(
    key: torch.Tensor,
    value: torch.Tensor,
    log4_n: int,
    g: int = 0,
    delta: float = 0.5,
    alias: str = "khcompress",
    subsample_method: str = "stride",
    M_inner: int = 64,
    softmax_temp: float = 1.0,
    symmetrize: bool = True,
    use_vbar_kbar: bool = False,
) -> torch.Tensor:
    """KH-Compress

    Computes a coreset of size n_out by first standard thinning the input sequence to size n,
    then calling the _khcompress function.

    Args:
        key: Tensor of shape [B, S, H, E]
        value: Tensor of shape [B, S, H, D]
        log4_n: log base 4 of n
        g: Oversampling factor, int >= 0
        delta: Kernel halving failure parameter, scalar in [0,1]
        alias: alias for the event
        subsample_method: subsampling method, one of "stride", "offset", "uniform", or "stratified"
        M_inner: inner dimension for block halve
        softmax_temp: scalar, softmax temperature for the exp kernel
        symmetrize: if True, randomly choose between coreset and its complement
        use_vbar_kbar: if True, use vbar and kbar to center the keys and values

    Returns:
        key coreset Tensor of shape [B, n_out, H, E] and
        value coreset Tensor of shape [B, n_out, H, D]

    """
    B, S, H, E = key.shape
    _, _, _, D = value.shape
    n = 4**log4_n
    num_halving_rounds = int(log4_n - g)
    bucket_size = int(4 ** (g + 1))
    stride = S // n
    n_times_stride = n * stride

    assert key.is_contiguous(), "key must be contiguous"
    assert value.is_contiguous(), "value must be contiguous"

    # If n <= 2^g sqrt(n), no further thinning is needed
    # Equivalently, check if n < bucket_size
    if n < bucket_size:
        return key[:, 0:n_times_stride:stride], value[:, 0:n_times_stride:stride]

    # Compute KH-Compress base halving probability parameter
    halve_prob = delta / bucket_size / num_halving_rounds / n

    # Draw uniform random variables here
    uniforms = {}
    symmetrize_rvs = {}
    for i in range(num_halving_rounds):
        S_ = bucket_size * 2**i
        log_multiplier = 0.5 + math.log(2 * S_ / (halve_prob * S_ * S_))
        uniforms[i] = torch.empty(
            B, n // bucket_size // 4**i, H, S_ // 2, device=key.device
        ).uniform_(-log_multiplier, log_multiplier)
        if M_inner > 0:
            uniforms[i] = uniforms[i].transpose(1, 2).view(B * H, -1, S_ // 2)
        else:
            uniforms[i] = uniforms[i].transpose(1, 2).view(-1, S_ // 2)
        # import pdb; pdb.set_trace()
        if symmetrize:
            symmetrize_rvs[i] = torch.empty(
                1, device=key.device, dtype=torch.bool
            ).bernoulli_()
        else:
            symmetrize_rvs[i] = None

    # Collapse batch and head dimensions
    key = key.transpose(1, 2).view(B * H, S, E)
    value = value.transpose(1, 2).view(B * H, S, D)
    # Subsample the original key and value points by passing in a strided input index for the first halving round
    # NOTE: this is much more efficient than using a gather operation on the original key and value points
    input_indices = arange(0, S, device=key.device, dtype=torch.int32)[
        0:n_times_stride:stride
    ].repeat(B * H, 1)
    alias_khcompress = f"{alias}__khcompress"
    with multi_events(alias_khcompress):
        key, value = _khcompress(
            key,
            value,
            bucket_size,
            num_halving_rounds,
            halve_prob,
            symmetrize=symmetrize,
            alias=alias_khcompress,
            M_inner=M_inner,
            input_indices=input_indices,
            softmax_temp=softmax_temp,
            use_vbar_kbar=use_vbar_kbar,
            uniforms=uniforms,
            symmetrize_rvs=symmetrize_rvs,
        )
    # import pdb; pdb.set_trace()

    # Un-collapse batch and head dimensions
    key = key.view(B, H, -1, E).transpose(1, 2)
    value = value.view(B, H, -1, D).transpose(1, 2)

    return key, value
