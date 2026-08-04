import heapq
from typing import Any, Iterator
import numpy as np


Number = np.float32


class MaxHeap:
    """Maintain top N highest scoring items in heap, useful for top N search results."""
    heap: list[tuple[Number, Any]]
    max_size: int

    def __init__(self,
                 max_size: int,
                 heap: list[tuple[Number, Any]] | None = None) -> None:
        if heap is None:
            self.heap = []
        else:
            if len(heap) > max_size:
                raise ValueError("Initial heap size cannot exceed max_size")
            self.heap = heap
        self.max_size = max_size

    def pushpop(self, item: tuple[Number, Any]) -> tuple[Number, Any] | None:
        """Push an item onto the heap and pop the smallest item if the heap exceeds max_size."""
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
        else:
            popped = heapq.heappushpop(self.heap, item)
            return popped

    @property
    def sorted(self) -> list[tuple[Number, Any]]:
        """Return the items in the heap sorted by score in descending order."""
        return sorted(self.heap, key=lambda x: x[0], reverse=True)

    def items(self) -> Iterator[tuple[Number, Any]]:
        """Return an iterator over the items in the heap."""
        return iter(self.heap)

    def __len__(self) -> int:
        return len(self.heap)

    def __bool__(self) -> bool:
        return bool(self.heap)

    def __iter__(self) -> Iterator[tuple[Number, Any]]:
        return iter(self.heap)

    def __repr__(self) -> str:
        return f"MaxHeap(max_size={self.max_size}, items={self.sorted})"

    def __str__(self) -> str:
        return f"MaxHeap(max_size={self.max_size}, items={self.sorted})"

    def __contains__(self, item: tuple[Number, Any]) -> bool:
        return item in self.heap

    def __getitem__(self, index: int) -> tuple[Number, Any]:
        return self.heap[index]
