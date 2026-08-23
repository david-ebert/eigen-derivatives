from collections.abc import Sequence
from typing import Any

class DerivativesList:
    """
    Container class for storing and evaluating function derivatives.

    Designed to handle sequences of NumPy arrays and SciPy sparse matrices with initial element being the function
    evaluation and the following its derivatives at the same parameter point.
    """
    __slots__ = ("_list", "_shape", "_size", "_ndim")

    def __init__(self, x: Sequence[Any]) -> None:
        if not x:
            raise ValueError("The derivatives list cannot be empty.")

        self._list = list(x)

        first_elem = self._list[0]
        if not hasattr(first_elem, "shape") or not hasattr(first_elem, "ndim"):
            raise TypeError("Elements must be NumPy arrays or SciPy sparse matrices.")

        self._shape: tuple[int, ...] = first_elem.shape
        self._size: int | None = getattr(first_elem, "size", None)
        self._ndim: int = first_elem.ndim

        if not all(elem.shape == self._shape for elem in self._list):
            raise ValueError("All derivative elements must have the exact same shape.")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int | None:
        return self._size

    @property
    def ndim(self) -> int:
        return self._ndim

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, item: int) -> Any:
        return self._list[item]

    def __repr__(self) -> str:
        return f"Derivatives list with length {len(self)}, shape {self.shape}."

    def __str__(self) -> str:
        def _ord_suffix(n: int) -> str:
            if 11 <= (n % 100) <= 13:
                return f"{n}th"
            return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

        lines = [f"Evaluation:\n{self._list[0]}"]
        for i, deriv in enumerate(self._list[1:], start=1):
            lines.append(f"{_ord_suffix(i)} derivative:\n{deriv}")

        return f"{repr(self)}\n" + "\n".join(lines)

    def evaluate_taylor(self, dx: float) -> Any:
        max_order = len(self) - 1
        accumulator = self._list[max_order].copy()

        for order in range(max_order - 1, -1, -1):
            accumulator = accumulator * (dx / (order + 1)) + self._list[order]

        return accumulator
