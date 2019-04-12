"""
Implementation of common data structures used.
"""
from math import sqrt, trunc
from typing import Optional, List, Iterator, Generic, TypeVar, Tuple
from enum import Enum

# Useless to add skip pointers if interval = 1 or 2
SKIP_INTERVAL_THRESHOLD = 3

T = TypeVar('T')


class Node(Generic[T]):
    """
    Represents a node in the linked list.
    You can access the value through attribute `value`,
    You can obtain the next node through the method `next()` or
    the skip node through the method `skip()`.
    `next()` and `skip()` will return `None` if there is no next node or
    skip node respectively.
    """

    def __init__(self, index: int, linked_list: 'LinkedList[T]') -> None:
        # pylint: disable=protected-access
        self.value = linked_list._data[index][0]
        self._index = index
        self._linked_list = linked_list

    def next(self) -> Optional['Node[T]']:
        """
        Gets the node after this node, or `None` if there is none.
        """
        # pylint: disable=protected-access
        return self._linked_list._get_next(self._index)

    def skip(self) -> Optional['Node[T]']:
        """
        Gets the skip node of this node, or `None` if there is none.
        """
        # pylint: disable=protected-access
        return self._linked_list._get_skip(self._index)


class LinkedList(Generic[T]):
    """
    A linked list implementation with skip pointers,
    backed by python's list for performance.
    """

    def __init__(self) -> None:
        self._data: List[Tuple[T, Optional[int]]] = []

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return self.__len__() > 0

    def __iter__(self) -> Iterator[T]:
        for value, _ in self._data:
            yield value

    def __str__(self) -> str:
        return " ".join(str(doc_id) for doc_id in self)

    def append(self, value: T) -> None:
        """
        Adds a new value to the tail of this linked list.
        """
        self._data.append((value, None))

    def extend(self, values: List[T]) -> None:
        """
        Extends the linked list by appending all the items from the iterable.
        """
        self._data.extend((value, None) for value in values)

    def get_head(self) -> Optional[Node[T]]:
        """
        Gets the head node of this linked list.
        """
        if not self._data:
            return None
        return Node(0, self)

    def build_skips(self) -> None:
        """
        Builds the skip pointer in this linked list, whereby sqrt(n) skip
        pointers are placed evenly, n = length of the list.
        """
        length = len(self)
        total_skips = trunc(sqrt(length))
        # Early bail if 0 skips
        if total_skips == 0:
            return
        interval = (length - 1) // total_skips
        # Early bail if interval is below the threshold
        if interval < SKIP_INTERVAL_THRESHOLD:
            return
        prev = 0
        for i in range(interval, total_skips * interval + 1, interval):
            value, _ = self._data[prev]
            self._data[prev] = (value, i)
            prev = i

    def _get_next(self, index: int) -> Optional[Node[T]]:
        if index >= len(self) - 1:
            return None
        return Node(index + 1, self)

    def _get_skip(self, index: int) -> Optional[Node[T]]:
        _, skip = self._data[index]
        if skip is None:
            return None
        return Node(skip, self)


class QueryType(Enum):
    BOOLEAN = 'boolean'
    FREE_TEXT = 'freetext'


class TokenType(Enum):
    PHRASE = 'phrase'
    NON_PHRASE = 'nonphrase'
