from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from eigen_derivatives._types import Matrix


class Backend(ABC):
    """Matrix construction and linear solves for one element type.

    Subclass this to support a further array library. Only the abstract methods
    are required; `factorize` has a working default.
    """

    @abstractmethod
    def as_matrix(self, matrix: Matrix) -> Matrix:
        """Return the argument in this backend's matrix type."""

    @abstractmethod
    def block(self, blocks: Sequence[Sequence[Matrix]]) -> Matrix:
        """Assemble a matrix from a nested sequence of blocks."""

    @abstractmethod
    def zeros(self, rows: int, cols: int) -> Matrix:
        """Return a zero matrix of the given shape."""

    @abstractmethod
    def eye(self, dim: int) -> Matrix:
        """Return an identity matrix of the given size."""

    @abstractmethod
    def solve(self, matrix: Matrix, rhs: np.ndarray) -> np.ndarray:
        """Solve a single linear system, returning a solution shaped like rhs."""

    def factorize(self, matrix: Matrix) -> Callable[[np.ndarray], np.ndarray]:
        """Return a solver that reuses one factorization across right-hand sides.

        The returned callable must accept a one-dimensional right-hand side or a
        matrix of them and return a solution of the same shape, which is what an
        overriding implementation has to preserve.

        The default performs no reuse, so a backend without a factorization is
        slower but still complete.
        """
        return partial(self.solve, matrix)


class DenseBackend(Backend):
    """NumPy arrays, with the LU factorization from SciPy."""

    def as_matrix(self, matrix: Matrix) -> np.ndarray:
        return np.asarray(matrix)

    def block(self, blocks: Sequence[Sequence[Matrix]]) -> np.ndarray:
        return np.block(blocks)

    def zeros(self, rows: int, cols: int) -> np.ndarray:
        return np.zeros((rows, cols))

    def eye(self, dim: int) -> np.ndarray:
        return np.eye(dim)

    def solve(self, matrix: Matrix, rhs: np.ndarray) -> np.ndarray:
        return np.linalg.solve(matrix, rhs)

    def factorize(self, matrix: Matrix) -> Callable[[np.ndarray], np.ndarray]:
        lu_and_pivots = la.lu_factor(np.asarray(matrix))
        return lambda rhs: la.lu_solve(lu_and_pivots, rhs)


class SparseBackend(Backend):
    """SciPy sparse arrays in CSC format, with SuperLU factorizations."""

    def as_matrix(self, matrix: Matrix) -> sp.csc_array:
        return sp.csc_array(matrix)

    def block(self, blocks: Sequence[Sequence[Matrix]]) -> Matrix:
        return sp.block_array(blocks)

    def zeros(self, rows: int, cols: int) -> sp.csc_array:
        return sp.csc_array((rows, cols))

    def eye(self, dim: int) -> Matrix:
        return sp.eye_array(dim, format="csc")

    def solve(self, matrix: Matrix, rhs: np.ndarray) -> np.ndarray:
        # spsolve collapses a single-column right-hand side, splu does not
        return spla.spsolve(sp.csc_array(matrix), rhs).reshape(np.shape(rhs))

    def factorize(self, matrix: Matrix) -> Callable[[np.ndarray], np.ndarray]:
        lu = spla.splu(sp.csc_array(matrix))
        return lu.solve


def get_backend(element: Matrix) -> Backend:
    """Return the backend that matches the element type of a series."""
    return SparseBackend() if sp.issparse(element) else DenseBackend()
