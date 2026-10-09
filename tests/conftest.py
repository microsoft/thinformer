"""Shared fixtures for the offline Express test suite.

All tests require a CUDA device (the kernels are Triton). Autotuning over the
full 144-config space takes minutes per round shape, so the session pins a
single kernel config; this also makes outputs bitwise reproducible across runs
within the session.
"""

from collections.abc import Iterator

import pytest
import torch


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip the whole suite when no CUDA device is available."""
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="CUDA device required for Triton kernels")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def pinned_kernel_config() -> Iterator[None]:
    """Pin one autotune config so tests are fast and deterministic."""
    if not torch.cuda.is_available():
        yield
        return
    import triton

    import thinformer.triton.wtd_attn as wa

    saved = wa._attn_fwd.configs
    wa._attn_fwd.configs = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_stages=3, num_warps=4)
    ]
    yield
    wa._attn_fwd.configs = saved
