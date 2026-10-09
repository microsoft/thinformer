"""Cuda events for timing operations inside the Thinformer module."""

import torch
from collections import defaultdict
from contextlib import contextmanager


class Events:
    def __init__(self) -> None:
        self.events = defaultdict(
            lambda: {
                "start": torch.cuda.Event(enable_timing=True),
                "end": torch.cuda.Event(enable_timing=True),
            }
        )

    def __getitem__(self, name: str) -> dict:
        return self.events[name]

    def reset(self) -> None:
        """Reset the events dictionary."""
        self.events = defaultdict(
            lambda: {
                "start": torch.cuda.Event(enable_timing=True),
                "end": torch.cuda.Event(enable_timing=True),
            }
        )

    def items(self) -> list[tuple[str, dict]]:
        """Return the items of the events dictionary."""
        return self.events.items()

    def values(self) -> list[dict]:
        """Return the values of the events dictionary."""
        return self.events.values()

    def keys(self) -> list[str]:
        """Return the keys of the events dictionary."""
        return self.events.keys()


# Initialize wrapper for cuda events dictionary
events = Events()


class MultiEvents:
    """Functionality to collect multiple events for the same key."""

    def __init__(self) -> None:
        self.events = defaultdict(list)

    def __getitem__(self, name: str):
        return self.events[name]

    def reset(self) -> None:
        """Reset the events dictionary."""
        self.events = defaultdict(list)

    def items(self) -> list[tuple[str, list]]:
        """Return the items of the events dictionary."""
        return self.events.items()

    def values(self) -> list[list]:
        """Return the values of the events dictionary."""
        return self.events.values()

    def keys(self) -> list[str]:
        """Return the keys of the events dictionary."""
        return self.events.keys()

    @contextmanager
    def __call__(self, name: str):
        """Context manager that automatically calls start and end for the given key."""
        # Initialize start and end events before yielding
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self.events[name].append((start, end))
        start.record()
        try:
            yield
        finally:
            end.record()


# Initialize wrapper for cuda events dictionary
multi_events = MultiEvents()
