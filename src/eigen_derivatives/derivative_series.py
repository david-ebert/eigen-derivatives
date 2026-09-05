from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Self

import numpy as np


@dataclass(frozen=True, eq=False, slots=True)
class DerivativeSeries:
    """
    Container class for storing and evaluating function derivatives.

    Designed to handle sequences of NumPy arrays and SciPy sparse matrices with initial element being the function
    evaluation and the following its derivatives at the same parameter point.

    The container itself is immutable: neither the tuple nor the attribute can be
    rebound after construction. The stored arrays are not copied, so mutating them
    in place is possible and remains the caller's responsibility.
    """

    derivatives: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "derivatives", tuple(self.derivatives))

        if not self.derivatives:
            raise ValueError("The derivative series cannot be empty.")

        first_elem = self.derivatives[0]
        if not hasattr(first_elem, "shape") or not hasattr(first_elem, "ndim"):
            raise TypeError("Elements must be NumPy arrays or SciPy sparse matrices.")

        if any(type(elem) is not type(first_elem) for elem in self.derivatives):
            found = sorted({type(elem).__name__ for elem in self.derivatives})
            raise TypeError(f"All elements must have the same type, got {' and '.join(found)}.")

        if any(elem.shape != first_elem.shape for elem in self.derivatives):
            raise ValueError("All derivative elements must have the exact same shape.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape shared by the evaluation and all derivatives."""
        return self.derivatives[0].shape

    def truncate(self, max_order: int) -> Self:
        """Return a new series with the evaluation and the derivatives up to max_order."""
        if not 0 <= max_order <= len(self) - 1:
            raise ValueError(
                f"max_order {max_order} is outside the available range 0 to {len(self) - 1}."
            )
        return type(self)(self.derivatives[:max_order + 1])

    def pad_with_zeros(self, max_order: int) -> Self:
        """Return a new series extended with zero derivatives up to max_order.

        Use this only when the higher derivatives really vanish. It states that they are
        zero, it does not stand in for derivatives the data does not provide. To drop
        orders that are not determined, use truncate.
        """
        if max_order < len(self) - 1:
            raise ValueError(
                f"max_order {max_order} is below the highest order {len(self) - 1} of this series; "
                "use truncate to shorten it."
            )
        return type(self)(self.derivatives + tuple(self._zero() for _ in range(max_order + 1 - len(self))))

    def _zero(self) -> Any:
        first_elem = self.derivatives[0]
        if hasattr(first_elem, "eliminate_zeros"):
            zero = first_elem * 0
            zero.eliminate_zeros()
            return zero
        return np.zeros_like(first_elem)

    def __len__(self) -> int:
        return len(self.derivatives)

    def __getitem__(self, item: int) -> Any:
        return self.derivatives[item]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.derivatives)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        raise TypeError("DerivativeSeries is not an array. Index it first, e.g. series[k].")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.derivatives!r})"

    def __str__(self) -> str:
        def _ord_suffix(n: int) -> str:
            if 11 <= (n % 100) <= 13:
                return f"{n}th"
            return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

        lines = [f"Evaluation:\n{self.derivatives[0]}"]
        for i, deriv in enumerate(self.derivatives[1:], start=1):
            lines.append(f"{_ord_suffix(i)} derivative:\n{deriv}")

        return "\n".join(lines)

    def evaluate_taylor(self, dx: float) -> Any:
        max_order = len(self) - 1
        dtype = np.result_type(*(elem.dtype for elem in self.derivatives), 1.0, dx)
        accumulator = self.derivatives[max_order].astype(dtype)  # astype copies

        for order in range(max_order - 1, -1, -1):
            accumulator *= dx / (order + 1)
            accumulator += self.derivatives[order]

        return accumulator
