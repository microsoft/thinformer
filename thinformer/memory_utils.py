"""Python class for keeping track of memory usage"""

import torch

# from collections import defaultdict
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Memory profiler that only records usage when it's explicitly enabled."""

    def __init__(self):
        self.usage = {}
        self.is_enabled = False
        logger.info("To enable memory profiling, call `profiler.enable()`")

    def enable(self) -> None:
        """Enable the memory profiler."""
        self.is_enabled = True
        logger.info("Memory profiler enabled.")
        logger.warning("NOTE: runtime will be affected by memory profiling.")

    @contextmanager
    def __call__(self, name: str, disable: bool = False):
        """Context manager to profile memory usage.

        Args:
            name (str): Name of the profiling context.
            disable (bool): If True, disable profiling for this context.
                NOTE: this overrides the global profiler state.

        """
        if not self.is_enabled or disable:
            yield
            return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # NOTE: synchronization here does not seems to change the amount of memory used
        # torch.cuda.synchronize()
        try:
            yield
        finally:
            self.usage[name] = torch.cuda.max_memory_allocated()

    def reset(self) -> None:
        """Reset the usage dictionary."""
        self.usage = {}

    def items(self) -> list[tuple[str, dict]]:
        """Return the items of the memory dictionary."""
        return self.usage.items()

    def values(self) -> list[dict]:
        """Return the values of the memory dictionary."""
        return self.usage.values()

    def keys(self) -> list[str]:
        """Return the keys of the memory dictionary."""
        return self.usage.keys()


# Initialize the profiler
profiler = MemoryProfiler()
