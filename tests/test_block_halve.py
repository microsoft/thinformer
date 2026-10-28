"""Tests for block halve

Example usage:
1. python:
```bash
RUN_TEST=1 python block_halve_test.py -s 1 -op out-0831 -bs 1 -mi 64
```

2. pytest
```bash
python -m pytest block_halve_test.py::test_op_no_softmax_temp_1 -v
```

Note:
- Add TRITON_INTERPRET=1 to use pdb breakpoints. Caveat: using this flag with pytest causes the script to hang.

"""

# %%
# Import torch reference implementation
from halve_reference import halve

# Import Triton implementation
from thinformer.triton.block_halve import block_halve
from halve_reference import is_hip_cdna2
import math
import os

# %%
import torch
import pytest
import triton

from argparse import ArgumentParser

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = getattr(torch, os.environ.get("DTYPE", "float16"))

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument("--output_path", "-op", type=str, default="out")
parser.add_argument("--batch_size", "-bs", type=int, default=1)
parser.add_argument("--dimension", "-d", type=int, default=64)
parser.add_argument("--seed", "-s", type=int, default=1)
parser.add_argument("--M_inner", "-mi", type=int, default=64)
args, _ = parser.parse_known_args()
B = args.batch_size
D = args.dimension
seed = args.seed
M_inner = args.M_inner

output_dir = args.output_path
os.makedirs(output_dir, exist_ok=True)
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)


# %%
@pytest.mark.parametrize("M", [2**i * M_inner for i in range(8)])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("symmetrize", [False])
def test_op_no_softmax_temp_1(M: int, seed: int, symmetrize: bool) -> None:
    """To run this test from command line, use:
    python -m pytest block_halve_test.py::test_op_no_softmax_temp_1 -v

    Args:
        M: int, the number of input points
        seed: int, the seed for the random number generator
        symmetrize: bool, if True, randomly choose between coreset and its complement

    """
    torch.manual_seed(seed)
    print(f"B={B}, M={M}, M_inner={M_inner}")
    sqrt_softmax_temp = math.sqrt(math.sqrt(1 / 64))
    key = torch.randn((B, M, D), device=DEVICE, dtype=DTYPE) * sqrt_softmax_temp
    value = torch.randn((B, M, D), device=DEVICE, dtype=DTYPE)
    shift = torch.amax(torch.square(key).sum(dim=2, keepdim=True), dim=1, keepdim=True)
    b_sqd = (
        torch.linalg.vector_norm(value, dim=(1, 2), ord=torch.inf, keepdim=True) ** 2
    )
    # Let's simulate calling Compress on n = B * M input points with 4^{g+1} = M
    # and one halving round
    halve_prob = 0.5 / M / 1 / (M * B)

    if symmetrize:
        symmetrize_rv = torch.empty(1, device=DEVICE, dtype=torch.bool).bernoulli_()
    else:
        symmetrize_rv = None

    torch.manual_seed(seed)
    input_indices = torch.arange(0, M, device=DEVICE, dtype=torch.int32)
    triton_Kdiff, triton_coreset_sum_diff, triton_coreset_indices = block_halve(
        key,
        key.mT,
        value,
        value.mT,
        halve_prob,
        b_sqd,
        M_inner=M_inner,
        # repeat input_coreset_indices B x 1 times so that we have a (B, 1, M) tensor
        input_indices=input_indices.repeat(B, 1, 1),
        symmetrize=symmetrize,
        symmetrize_rv=symmetrize_rv,
    )
    triton_o_key = key.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, D).to(torch.int64)
    )
    triton_o_value = value.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, D).to(torch.int64)
    )

    torch.manual_seed(seed)
    torch_Kdiff, torch_coreset_indices, torch_o_key, torch_o_value = halve(
        key,
        key.mT,
        value,
        value.mT,
        # block_halve treats halve_prob as a total probability (1b8ac81) while
        # halve() uses delta = halve_prob * n^2; rescale so both draw the same uniforms
        halve_prob / M ** 2,
        shift,
        b_sqd,
        symmetrize_rv=symmetrize_rv,
    )

    rtol = 1e-2 if is_hip_cdna2() else 0
    torch.testing.assert_close(triton_o_key, torch_o_key, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(triton_o_value, torch_o_value, atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("M", [2**i * M_inner for i in range(8)])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("symmetrize", [False])
def test_op_no_softmax_temp_2(M: int, seed: int, symmetrize: bool) -> None:
    """To run this test from command line, use:
    python -m pytest block_halve_test.py::test_op_no_softmax_temp_2 -v

    Args:
        M: int, the number of input points
        seed: int, the seed for the random number generator
        symmetrize: bool, if True, randomly choose between coreset and its complement

    """
    torch.manual_seed(seed)
    print(f"B={B}, M={M}, M_inner={M_inner}")
    sqrt_softmax_temp = math.sqrt(math.sqrt(1 / 64))
    key = torch.randn((B, M, 64), device=DEVICE, dtype=DTYPE) * sqrt_softmax_temp
    value = torch.randn((B, M, 64), device=DEVICE, dtype=DTYPE)
    shift = torch.amax(torch.square(key).sum(dim=2, keepdim=True), dim=1, keepdim=True)
    b_sqd = (
        torch.linalg.vector_norm(value, dim=(1, 2), ord=torch.inf, keepdim=True) ** 2
    )
    # Let's simulate calling Compress on n = B * M input points with 4^{g+1} = M//2
    # and one halving round
    halve_prob = 0.5 / (M // 2) / 1 / (M * B)

    torch.manual_seed(seed)
    input_indices = torch.arange(0, M, device=DEVICE, dtype=torch.int32)
    triton_Kdiff, triton_coreset_sum_diff, triton_coreset_indices = block_halve(
        key,
        key.mT,
        value,
        value.mT,
        halve_prob,
        b_sqd,
        M_inner=M_inner,
        # repeat input_coreset_indices B x 1 times so that we have a (B, 1, M) tensor
        # then reshape to (B, 2, M//2)
        input_indices=input_indices.repeat(B, 1, 1).view(B, 2, -1),
        symmetrize=symmetrize,
    )
    triton_o_key = key.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )
    triton_o_value = value.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )

    torch.manual_seed(seed)
    torch_Kdiff, torch_coreset_indices, torch_o_key, torch_o_value = halve(
        key[:, input_indices].view(B * 2, -1, 64),
        key[:, input_indices].view(B * 2, -1, 64).mT,
        value[:, input_indices].view(B * 2, -1, 64),
        value[:, input_indices].view(B * 2, -1, 64).mT,
        # block_halve treats halve_prob as a total probability (1b8ac81) while
        # halve() uses delta = halve_prob * n^2; rescale so both draw the same uniforms
        halve_prob / (M // 2) ** 2,
        shift,
        b_sqd,
    )
    torch_o_key = torch_o_key.view(B, -1, 64)
    torch_o_value = torch_o_value.view(B, -1, 64)

    rtol = 1e-2 if is_hip_cdna2() else 0
    torch.testing.assert_close(triton_o_key, torch_o_key, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(triton_o_value, torch_o_value, atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("M", [2**i * M_inner for i in range(8)])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("symmetrize", [False])
def test_op_softmax_temp_1(M: int, seed: int, symmetrize: bool) -> None:
    """To run this test from command line, use:
    python -m pytest block_halve_test.py::test_op_softmax_temp_1 -v

    Args:
        M: int, the number of input points
        seed: int, the seed for the random number generator
        symmetrize: bool, if True, randomly choose between coreset and its complement

    """
    torch.manual_seed(seed)
    print(f"B={B}, M={M}, M_inner={M_inner}")
    softmax_temp = math.sqrt(1 / 64)
    key = torch.randn((B, M, 64), device=DEVICE, dtype=DTYPE)
    value = torch.randn((B, M, 64), device=DEVICE, dtype=DTYPE)
    shift = torch.amax(torch.square(key).sum(dim=2, keepdim=True), dim=1, keepdim=True)
    b_sqd = (
        torch.linalg.vector_norm(value, dim=(1, 2), ord=torch.inf, keepdim=True) ** 2
    )
    # Let's simulate calling Compress on n = B * M input points with 4^{g+1} = M and one halving round
    halve_prob = 0.5 / M / 1 / (M * B)

    torch.manual_seed(seed)
    input_indices = torch.arange(0, M, device=DEVICE, dtype=torch.int32)
    triton_Kdiff, triton_coreset_sum_diff, triton_coreset_indices = block_halve(
        key,
        key.mT,
        value,
        value.mT,
        halve_prob,
        b_sqd,
        M_inner=M_inner,
        # repeat input_coreset_indices B x 1 times so that we have a (B, 1, M) tensor
        input_indices=input_indices.repeat(B, 1, 1),
        softmax_temp=softmax_temp,
        symmetrize=symmetrize,
    )
    triton_o_key = key.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )
    triton_o_value = value.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )

    torch.manual_seed(seed)
    torch_Kdiff, torch_coreset_indices, torch_o_key, torch_o_value = halve(
        key,
        key.mT,
        value,
        value.mT,
        # block_halve treats halve_prob as a total probability (1b8ac81) while
        # halve() uses delta = halve_prob * n^2; rescale so both draw the same uniforms
        halve_prob / M ** 2,
        shift,
        b_sqd,
        softmax_temp=softmax_temp,
    )

    rtol = 1e-2 if is_hip_cdna2() else 0
    torch.testing.assert_close(triton_o_key, torch_o_key, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(triton_o_value, torch_o_value, atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("M", [2**i * M_inner for i in range(8)])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("symmetrize", [False])
def test_op_softmax_temp_2(M: int, seed: int, symmetrize: bool) -> None:
    """To run this test from command line, use:
    python -m pytest block_halve_test.py::test_op_softmax_temp_2 -v

    Args:
        M: int, the number of input points
        seed: int, the seed for the random number generator
        symmetrize: bool, if True, randomly choose between coreset and its complement

    """
    torch.manual_seed(seed)
    print(f"B={B}, M={M}, M_inner={M_inner}")
    softmax_temp = math.sqrt(1 / 64)
    key = torch.randn((B, M, 64), device=DEVICE, dtype=DTYPE)
    value = torch.randn((B, M, 64), device=DEVICE, dtype=DTYPE)
    # NOTE: the reference implementation uses a single shift per batch,
    # while the Triton implementation uses a shift per row of Kdiff
    shift = torch.amax(torch.square(key).sum(dim=2, keepdim=True), dim=1, keepdim=True)
    b_sqd = (
        torch.linalg.vector_norm(value, dim=(1, 2), ord=torch.inf, keepdim=True) ** 2
    )
    # Let's simulate calling Compress on n = B * M input points with 4^{g+1} = M//2 and one halving round
    halve_prob = 0.5 / (M // 2) / 1 / (M * B)

    torch.manual_seed(seed)
    input_indices = torch.arange(0, M, device=DEVICE, dtype=torch.int32)
    triton_Kdiff, triton_coreset_sum_diff, triton_coreset_indices = block_halve(
        key,
        key.mT,
        value,
        value.mT,
        halve_prob,
        b_sqd,
        M_inner=M_inner,
        # repeat input_coreset_indices B x 1 times so that we have a (B, 1, M) tensor
        input_indices=input_indices.repeat(B, 1, 1).view(B, 2, -1),
        softmax_temp=softmax_temp,
        symmetrize=symmetrize,
    )
    triton_o_key = key.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )
    triton_o_value = value.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )

    torch.manual_seed(seed)
    torch_Kdiff, torch_coreset_indices, torch_o_key, torch_o_value = halve(
        key[:, input_indices].view(B * 2, -1, 64),
        key[:, input_indices].view(B * 2, -1, 64).mT,
        value[:, input_indices].view(B * 2, -1, 64),
        value[:, input_indices].view(B * 2, -1, 64).mT,
        # block_halve treats halve_prob as a total probability (1b8ac81) while
        # halve() uses delta = halve_prob * n^2; rescale so both draw the same uniforms
        halve_prob / (M // 2) ** 2,
        shift,
        b_sqd,
        softmax_temp=softmax_temp,
    )
    torch_o_key = torch_o_key.view(B, -1, 64)
    torch_o_value = torch_o_value.view(B, -1, 64)

    rtol = 1e-2 if is_hip_cdna2() else 0
    torch.testing.assert_close(triton_o_key, torch_o_key, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(triton_o_value, torch_o_value, atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("M", [2**i * M_inner for i in range(8)])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("symmetrize", [False])
def test_op_softmax_temp_vbar_kbar_1(M: int, seed: int, symmetrize: bool) -> None:
    """To run this test from command line, use:
    python -m pytest block_halve_test.py::test_op_softmax_temp_vbar_kbar_1 -v

    Args:
        M: int, the number of input points
        seed: int, the seed for the random number generator
        symmetrize: bool, if True, randomly choose between coreset and its complement

    """
    torch.manual_seed(seed)
    print(f"B={B}, M={M}, M_inner={M_inner}")
    softmax_temp = math.sqrt(1 / 64)
    key = torch.randn((B, M, 64), device=DEVICE, dtype=DTYPE)
    value = torch.randn((B, M, 64), device=DEVICE, dtype=DTYPE)

    vmax, _ = torch.max(value, dim=1, keepdim=True)
    vmin, _ = torch.min(value, dim=1, keepdim=True)
    vbar = 0.5 * (vmax + vmin)
    kbar = torch.mean(key, dim=1, keepdim=True)

    shift = torch.amax(
        torch.square(key - kbar).sum(dim=2, keepdim=True), dim=1, keepdim=True
    )
    b_sqd = (
        torch.linalg.vector_norm(value - vbar, dim=(1, 2), ord=torch.inf, keepdim=True)
        ** 2
    )

    # Let's simulate calling Compress on n = B * M input points with 4^{g+1} = M and one halving round
    halve_prob = 0.5 / M / 1 / (M * B)

    torch.manual_seed(seed)
    input_indices = torch.arange(0, M, device=DEVICE, dtype=torch.int32)
    triton_Kdiff, triton_coreset_sum_diff, triton_coreset_indices = block_halve(
        key - kbar,
        (key - kbar).mT,
        value - vbar,
        (value - vbar).mT,
        halve_prob,
        b_sqd,
        M_inner=M_inner,
        # repeat input_coreset_indices B x 1 times so that we have a (B, 1, M) tensor
        input_indices=input_indices.repeat(B, 1, 1),
        softmax_temp=softmax_temp,
        symmetrize=symmetrize,
    )
    triton_o_key = key.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )
    triton_o_value = value.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )

    torch.manual_seed(seed)
    torch_Kdiff, torch_coreset_indices, torch_o_key, torch_o_value = halve(
        key,
        key.mT,
        value,
        value.mT,
        # block_halve treats halve_prob as a total probability (1b8ac81) while
        # halve() uses delta = halve_prob * n^2; rescale so both draw the same uniforms
        halve_prob / M ** 2,
        shift,
        b_sqd,
        softmax_temp=softmax_temp,
        vbar=vbar,
        kbar=kbar,
    )

    rtol = 1e-2 if is_hip_cdna2() else 0
    torch.testing.assert_close(triton_o_key, torch_o_key, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(triton_o_value, torch_o_value, atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("M", [2**i * M_inner for i in range(8)])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("symmetrize", [False])
def test_op_softmax_temp_vbar_kbar_2(M: int, seed: int, symmetrize: bool) -> None:
    """To run this test from command line, use:
    python -m pytest block_halve_test.py::test_op_softmax_temp_vbar_kbar_2 -v

    Args:
        M: int, the number of input points
        seed: int, the seed for the random number generator
        symmetrize: bool, if True, randomly choose between coreset and its complement

    """
    torch.manual_seed(seed)
    print(f"B={B}, M={M}, M_inner={M_inner}")
    softmax_temp = math.sqrt(1 / 64)
    key = torch.randn((B, M, 64), device=DEVICE, dtype=DTYPE)
    value = torch.randn((B, M, 64), device=DEVICE, dtype=DTYPE)

    vmax, _ = torch.max(value, dim=1, keepdim=True)
    vmin, _ = torch.min(value, dim=1, keepdim=True)
    vbar = 0.5 * (vmax + vmin)
    kbar = torch.mean(key, dim=1, keepdim=True)

    shift = torch.amax(
        torch.square(key - kbar).sum(dim=2, keepdim=True), dim=1, keepdim=True
    )
    b_sqd = (
        torch.linalg.vector_norm(value - vbar, dim=(1, 2), ord=torch.inf, keepdim=True)
        ** 2
    )
    # Let's simulate calling Compress on n = B * M input points with 4^{g+1} = M//2 and one halving round
    halve_prob = 0.5 / (M // 2) / 1 / (M * B)

    torch.manual_seed(seed)
    input_indices = torch.arange(0, M, device=DEVICE, dtype=torch.int32)
    triton_Kdiff, triton_coreset_sum_diff, triton_coreset_indices = block_halve(
        key - kbar,
        (key - kbar).mT,
        value - vbar,
        (value - vbar).mT,
        halve_prob,
        b_sqd,
        M_inner=M_inner,
        # repeat input_coreset_indices B x 1 times so that we have a (B, 1, M) tensor
        input_indices=input_indices.repeat(B, 1, 1).view(B, 2, -1),
        softmax_temp=softmax_temp,
        symmetrize=symmetrize,
    )
    triton_o_key = key.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )
    triton_o_value = value.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )

    torch.manual_seed(seed)
    torch_Kdiff, torch_coreset_indices, torch_o_key, torch_o_value = halve(
        key[:, input_indices].view(B * 2, -1, 64),
        key[:, input_indices].view(B * 2, -1, 64).mT,
        value[:, input_indices].view(B * 2, -1, 64),
        value[:, input_indices].view(B * 2, -1, 64).mT,
        # block_halve treats halve_prob as a total probability (1b8ac81) while
        # halve() uses delta = halve_prob * n^2; rescale so both draw the same uniforms
        halve_prob / (M // 2) ** 2,
        shift,
        b_sqd,
        softmax_temp=softmax_temp,
        vbar=torch.repeat_interleave(vbar, 2 * B // vbar.shape[0], dim=0),
        kbar=torch.repeat_interleave(kbar, 2 * B // kbar.shape[0], dim=0),
    )
    torch_o_key = torch_o_key.view(B, -1, 64)
    torch_o_value = torch_o_value.view(B, -1, 64)

    rtol = 1e-2 if is_hip_cdna2() else 0
    torch.testing.assert_close(triton_o_key, torch_o_key, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(triton_o_value, torch_o_value, atol=1e-2, rtol=rtol)


@pytest.mark.parametrize(
    "B, M, M_inner",
    [
        # M < M_inner: a single block is larger than the problem
        (4, 32, 64),
        (4, 64, 128),
        # B > 1, including the shapes in benchmarks/block_halve.py
        (1024, 64, 128),
        (256, 128, 64),
        (64, 256, 64),
        (8, 1024, 64),
    ],
)
@pytest.mark.parametrize("A", [1, 2])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_op_batch(B: int, M: int, M_inner: int, A: int, seed: int) -> None:
    """To run this test from command line, use:
    python -m pytest test_block_halve.py::test_op_batch -v

    Args:
        B: int, the batch size
        M: int, the number of input points per batch element
        M_inner: int, the block size used by block_halve
        A: int, the number of halving problems per batch element
        seed: int, the seed for the random number generator

    """
    torch.manual_seed(seed)
    sqrt_softmax_temp = math.sqrt(math.sqrt(1 / 64))
    key = torch.randn((B, M, D), device=DEVICE, dtype=DTYPE) * sqrt_softmax_temp
    value = torch.randn((B, M, D), device=DEVICE, dtype=DTYPE)
    shift = torch.amax(torch.square(key).sum(dim=2, keepdim=True), dim=1, keepdim=True)
    b_sqd = (
        torch.linalg.vector_norm(value, dim=(1, 2), ord=torch.inf, keepdim=True) ** 2
    )
    # Let's simulate calling Compress on n = B * M input points with 4^{g+1} = M//A
    # and one halving round
    halve_prob = 0.5 / (M // A) / 1 / (M * B)

    torch.manual_seed(seed)
    input_indices = torch.arange(0, M, device=DEVICE, dtype=torch.int32)
    triton_Kdiff, triton_coreset_sum_diff, triton_coreset_indices = block_halve(
        key,
        key.mT,
        value,
        value.mT,
        halve_prob,
        b_sqd,
        M_inner=M_inner,
        # repeat input_coreset_indices B x 1 times so that we have a (B, 1, M) tensor
        # then reshape to (B, A, M//A)
        input_indices=input_indices.repeat(B, 1, 1).view(B, A, -1),
        symmetrize=False,
    )
    triton_o_key = key.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, D).to(torch.int64)
    )
    triton_o_value = value.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, D).to(torch.int64)
    )

    torch.manual_seed(seed)
    torch_Kdiff, torch_coreset_indices, torch_o_key, torch_o_value = halve(
        key.view(B * A, -1, D),
        key.view(B * A, -1, D).mT,
        value.view(B * A, -1, D),
        value.view(B * A, -1, D).mT,
        # block_halve treats halve_prob as a total probability (1b8ac81) while
        # halve() uses delta = halve_prob * n^2; rescale so both draw the same uniforms
        halve_prob / (M // A) ** 2,
        shift,
        b_sqd,
    )
    torch_o_key = torch_o_key.view(B, -1, D)
    torch_o_value = torch_o_value.view(B, -1, D)

    rtol = 1e-2 if is_hip_cdna2() else 0
    torch.testing.assert_close(triton_o_key, torch_o_key, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(triton_o_value, torch_o_value, atol=1e-2, rtol=rtol)


@pytest.mark.parametrize("S", [20203])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("symmetrize", [False])
def test_op_softmax_temp_vbar_kbar_3(S: int, seed: int, symmetrize: bool) -> None:
    """To run this test from command line, use:
    python -m pytest block_halve_test.py::test_op_softmax_temp_vbar_kbar_3 -v

    Args:
        S: int, the number of input points
        seed: int, the seed for the random number generator
        symmetrize: bool, if True, randomly choose between coreset and its complement

    """
    torch.manual_seed(seed)
    print(f"B={B}, S={S}, M_inner={M_inner}")
    softmax_temp = math.sqrt(1 / 64)
    key = torch.randn((B, S, 64), device=DEVICE, dtype=DTYPE)
    value = torch.randn((B, S, 64), device=DEVICE, dtype=DTYPE)

    log2_nout = 11  # nout = 2048
    nin = 4**log2_nout // 2 // 2
    g = 0
    bucket_size = int(4 ** (g + 1)) * 2 * 2
    num_halving_rounds = log2_nout - 1 - 1

    # assert key.is_contiguous(), "key must be contiguous"
    # assert value.is_contiguous(), "value must be contiguous"
    # key = key.contiguous()
    # value = value.contiguous()

    # Compute KH-Compress base halving probability parameter
    halve_prob = 0.5 / bucket_size / num_halving_rounds / nin

    vmax, _ = torch.max(value, dim=1, keepdim=True)
    vmin, _ = torch.min(value, dim=1, keepdim=True)
    vbar = 0.5 * (vmax + vmin)
    kbar = torch.mean(key, dim=1, keepdim=True)

    # shift = torch.amax(
    #     torch.square(key - kbar).sum(dim=2, keepdim=True), dim=1, keepdim=True
    # )
    b_sqd = (
        torch.linalg.vector_norm(value - vbar, dim=(1, 2), ord=torch.inf, keepdim=True)
        ** 2
    )
    # Let's simulate calling Compress on n = B * M input points with 4^{g+1} = M//2 and one halving round
    # halve_prob = 0.5 / (M // 2) / 1 / (M * B)

    torch.manual_seed(seed)
    # Repeat the input indices instead of the key and value tensors
    a_floor = math.floor(nin / S)
    num_repeats = torch.ones(S, device=key.device, dtype=torch.int32) * a_floor
    remainder_vals = torch.rand(S, device=key.device)
    remainder_indices = torch.argsort(remainder_vals)[: nin % S]
    # Sort input indices to retain original order of points
    remainder_indices, _ = torch.sort(remainder_indices)
    # increment the number of repeats for the remainder indices
    num_repeats[remainder_indices] += 1
    # Repeat 0, 1, ..., S-1 num_repeats[0], num_repeats[1], ..., num_repeats[S-1] times
    input_indices = (
        torch.arange(0, S, device=key.device, dtype=torch.int32)
        .repeat(B, 1)
        .repeat_interleave(num_repeats, dim=1)
    )
    # import pdb; pdb.set_trace()
    _, _, triton_coreset_indices = block_halve(
        key - kbar,
        (key - kbar).mT,
        value - vbar,
        (value - vbar).mT,
        halve_prob,
        b_sqd,
        M_inner=M_inner,
        input_indices=input_indices.view(B, -1, bucket_size),
        softmax_temp=softmax_temp,
        symmetrize=symmetrize,
    )
    _ = key.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )
    _ = value.gather(
        1, triton_coreset_indices.flatten(1).unsqueeze(-1).expand(-1, -1, 64).to(torch.int64)
    )

    # torch.manual_seed(seed)
    # torch_Kdiff, torch_coreset_indices, torch_o_key, torch_o_value = halve(
    #     key[:, input_indices].view(B, -1, 64),
    #     key[:, input_indices].view(B, -1, 64).mT,
    #     value[:, input_indices].view(B, -1, 64),
    #     value[:, input_indices].view(B, -1, 64).mT,
    #     halve_prob,
    #     shift,
    #     b_sqd,
    #     softmax_temp=softmax_temp,
    #     vbar=torch.repeat_interleave(vbar, 2 * B // vbar.shape[0], dim=0),
    #     kbar=torch.repeat_interleave(kbar, 2 * B // kbar.shape[0], dim=0),
    # )
    # torch_o_key = torch_o_key.view(B, -1, 64)
    # torch_o_value = torch_o_value.view(B, -1, 64)

    # rtol = 1e-2 if is_hip_cdna2() else 0
    # torch.testing.assert_close(triton_o_key, torch_o_key, atol=1e-2, rtol=rtol)
    # torch.testing.assert_close(triton_o_value, torch_o_value, atol=1e-2, rtol=rtol)


# %%
# Exactness tests against float64 sequential halving.
#
# block_halve runs sequential kernel halving in blocks of M_inner points: the
# diagonal blocks are halved by _batch_halve_inner and each block's swap
# decisions are pushed to all later points by _batch_flash_psi. Every decision
# should match unblocked sequential halving,
#
#     swap_i = u_i <= psi_i,   psi_i = sum_{j < i} eta_j * Kdiff[j, i],
#
# where Kdiff is the pairwise-differenced kernel matrix and u_i is the uniform
# scaled by sqrt(Kdiff[i, i]) * max_{j <= i} sqrt(Kdiff[j, j]).
#
# The uniforms are drawn from a narrow range so that psi decides a sizeable
# fraction of swaps; with the default range (about +-3 log M) psi almost never
# changes a decision, so the reference comparisons above cannot detect psi
# errors. Decisions are checked with teacher forcing (psi is rebuilt from
# block_halve's own earlier decisions), so a single near-tie cannot cascade,
# and decisions whose float64 margin is within EXACT_TOL of the threshold scale
# are not judged.
EXACT_TOL = 1e-4


def make_inputs(
    B: int, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate seeded keys, values and b_sqd as in attention."""
    torch.manual_seed(seed)
    key = torch.randn(B, n, D, device=DEVICE, dtype=DTYPE) * D**-0.25
    value = torch.randn(B, n, D, device=DEVICE, dtype=DTYPE)
    b_sqd = (
        torch.linalg.vector_norm(value, dim=(1, 2), ord=torch.inf, keepdim=True) ** 2
    )
    return key, value, b_sqd


def count_wrong_decisions(
    key: torch.Tensor,
    value: torch.Tensor,
    b_sqd: float,
    uniforms: torch.Tensor,
    swaps: torch.Tensor,
) -> tuple[int, int]:
    """Check one halving problem against float64 sequential halving.

    Args:
        key: tensor of shape [M, E], the problem's keys in order
        value: tensor of shape [M, D], the problem's values in order
        b_sqd: scalar added to value inner products
        uniforms: tensor of shape [M // 2], unscaled uniforms
        swaps: bool tensor of shape [M // 2], block_halve's swap decisions

    Returns:
        (number of wrong decisions, number of decisions judged)

    """
    k, v = key.double(), value.double()
    sq = (k * k).sum(-1)
    K = torch.exp(k @ k.T - sq.max()) * (v @ v.T + b_sqd)
    Kdiff = K[:, ::2] - K[:, 1::2]
    Kdiff = Kdiff[::2] - Kdiff[1::2]
    rt_diag = Kdiff.diagonal().sqrt()
    scale = rt_diag * torch.cummax(rt_diag, 0).values
    thresh = uniforms.double() * scale
    eta = 1.0 - 2.0 * swaps.double()
    # psi_i = sum_{j < i} eta_j * Kdiff[j, i], using block_halve's decisions
    psi = (torch.triu(Kdiff, diagonal=1) * eta[:, None]).sum(0)
    expected = thresh <= psi
    judged = (thresh - psi).abs() > EXACT_TOL * scale
    judged[0] = False  # the first point never swaps
    wrong = (expected != swaps) & judged
    return int(wrong.sum()), int(judged.sum())


@pytest.mark.parametrize("M", [256, 1024, 4096])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("uniform_range", [0.1, 1.0])
def test_matches_sequential_halving(M: int, seed: int, uniform_range: float) -> None:
    """Single halving problem per batch row, spanning many M_inner blocks."""
    B = 2
    key, value, b_sqd = make_inputs(B, M, seed)
    uniforms = torch.empty(B, 1, M // 2, device=DEVICE).uniform_(
        -uniform_range, uniform_range
    )
    input_indices = torch.arange(M, device=DEVICE, dtype=torch.int32).repeat(B, 1, 1)
    _, _, coreset_indices = block_halve(
        key, key.mT, value, value.mT, 0.5 / M / (M * B), b_sqd,
        input_indices=input_indices, symmetrize=False, uniforms_buf=uniforms,
    )
    for b in range(B):
        swaps = (coreset_indices[b, 0] - input_indices[b, 0, ::2]).bool()
        wrong, judged = count_wrong_decisions(
            key[b], value[b], b_sqd[b].item(), uniforms[b, 0], swaps
        )
        assert judged > 0.9 * (M // 2 - 1)
        assert wrong == 0, f"batch {b}: {wrong}/{judged} decisions differ"


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_sequential_halving_gathered(seed: int) -> None:
    """Several halving problems per batch row over permuted input indices."""
    B, A, M = 2, 3, 1024
    n = A * M + 128
    key, value, b_sqd = make_inputs(B, n, seed)
    input_indices = torch.stack(
        [torch.randperm(n, device=DEVICE)[: A * M] for _ in range(B)]
    ).to(torch.int32).view(B, A, M)
    uniforms = torch.empty(B, A, M // 2, device=DEVICE).uniform_(-0.1, 0.1)
    _, _, coreset_indices = block_halve(
        key, key.mT, value, value.mT, 0.5 / M / (M * B), b_sqd,
        input_indices=input_indices, symmetrize=False, uniforms_buf=uniforms,
    )
    for b in range(B):
        for a in range(A):
            idx = input_indices[b, a].long()
            swaps = coreset_indices[b, a] == input_indices[b, a, 1::2]
            assert torch.all(
                swaps | (coreset_indices[b, a] == input_indices[b, a, ::2])
            )
            wrong, judged = count_wrong_decisions(
                key[b, idx], value[b, idx], b_sqd[b].item(), uniforms[b, a], swaps
            )
            assert judged > 0.9 * (M // 2 - 1)
            assert wrong == 0, (
                f"batch {b}, problem {a}: {wrong}/{judged} decisions differ"
            )


if __name__ == "__main__":
    test_cases = os.environ.get("RUN_TEST").split(",")
    for test_case in test_cases:
        match test_case:
            case "1":
                # test_op_no_softmax_temp_1(M=4, seed=2, symmetrize=False)
                # test_op_no_softmax_temp_1(M=62, seed=1, symmetrize=False)
                # test_op_no_softmax_temp_1(M=64, seed=1, symmetrize=False)
                # test_op_no_softmax_temp_1(M=66, seed=1, symmetrize=False)  # Failed on A6000
                # test_op_no_softmax_temp_1(M=126, seed=1, symmetrize=False)
                # test_op_no_softmax_temp_1(M=128, seed=1, symmetrize=False)
                # test_op_no_softmax_temp_1(M=130, seed=1, symmetrize=False)  # Failed on A6000
                # test_op_no_softmax_temp_1(M=190, seed=1, symmetrize=False)
                # test_op_no_softmax_temp_1(M=192, seed=1, symmetrize=False)
                # test_op_no_softmax_temp_1(M=194, seed=1, symmetrize=False)
                test_op_no_softmax_temp_1(M=3818, seed=1, symmetrize=False)
            case "2":
                test_op_no_softmax_temp_2(M=128, seed=1, symmetrize=False)
                # test_op_no_softmax_temp_2(M=4096, seed=3, symmetrize=False)
            case "3":
                test_op_softmax_temp_1(M=256, seed=1, symmetrize=False)
            case "4":
                # test_op_softmax_temp_2(M=128, seed=1, symmetrize=False)
                # The following test case is expected to fail on a 3090 GPU
                test_op_softmax_temp_2(M=4096, seed=3, symmetrize=False)
            case "5":
                test_op_softmax_temp_vbar_kbar_1(M=256, seed=1, symmetrize=False)
            case "6":
                test_op_softmax_temp_vbar_kbar_2(M=256, seed=1, symmetrize=False)
            case "7":
                test_op_softmax_temp_vbar_kbar_3(S=20203, seed=1, symmetrize=False)
            case _:
                raise ValueError(
                    f"Invalid RUN_TEST value: {os.environ.get('RUN_TEST')}"
                )
        print(f"✅ Passed test case: {test_case}")
