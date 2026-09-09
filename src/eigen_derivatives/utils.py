import itertools
from collections.abc import Iterator

import numpy as np
import scipy.sparse as sp

from eigen_derivatives._types import Matrix


def _get_numeric_backend(is_sparse: bool) -> tuple:
    """Return the function bindings and matrix factories for a sparse or a dense backend."""
    if is_sparse:
        return (
            sp.csc_array,
            sp.block_array,
            lambda system_matrix, rhs: sp.linalg.spsolve(system_matrix.tocsc(), rhs).reshape(np.shape(rhs)),
            lambda rows, cols: sp.csc_array((rows, cols)),
            lambda dim: sp.eye_array(dim, format="csc")
        )

    return (
        np.asarray,
        np.block,
        np.linalg.solve,
        lambda rows, cols: np.zeros((rows, cols)),
        lambda dim: np.eye(dim)
    )


def _multiindex_total_order(total_order: int, length: int) -> np.ndarray:
    """Return all multi-indices of a specific length and order."""
    if length <= 0:
        return np.empty((0, 0), dtype=int)
    if total_order < 0:
        return np.empty((0, length), dtype=int)
    if length == 1:
        return np.array([[total_order]], dtype=int)

    # stars and bars: a multi-index is a choice of length - 1 bar positions among the slots
    slots = total_order + length - 1
    bars = np.array(list(itertools.combinations(range(slots), length - 1)), dtype=int)
    borders = np.concatenate(
        [np.full((len(bars), 1), -1), bars, np.full((len(bars), 1), slots)], axis=1
    )
    return np.diff(borders, axis=1) - 1


def _multinomial_coefficient(list_of_multi_indices: np.ndarray) -> list[int]:
    """Return the multinomial coefficients of the multi-indices."""
    multi_indices = np.asarray(list_of_multi_indices, dtype=int)
    if multi_indices.ndim == 1:
        multi_indices = multi_indices.reshape(1, -1)
    if multi_indices.size == 0:
        return []

    # exact integer arithmetic, with the factorials tabulated once instead of per entry
    factorials = [1]
    for value in range(1, int(multi_indices.sum(axis=1).max()) + 1):
        factorials.append(factorials[-1] * value)

    coefficients = []
    for row in multi_indices.tolist():
        denominator = 1
        for entry in row:
            denominator *= factorials[entry]
        coefficients.append(factorials[sum(row)] // denominator)
    return coefficients


def _validate_mass_series(mass_mat_ds, num_orders: int) -> None:
    """Reject a mass series that is neither constant nor as long as the stiffness series."""
    if len(mass_mat_ds) not in (1, num_orders):
        raise ValueError(
            f"The mass series has {len(mass_mat_ds)} entries, expected 1 for a constant mass "
            f"matrix or {num_orders} to match the stiffness series. "
            f"Use pad_with_zeros({num_orders - 1}) to extend it."
        )


def _bilinear_form(
        left: np.ndarray,
        right: np.ndarray,
        *,
        middle: Matrix | None = None
) -> np.ndarray | np.generic:
    """Return left^H @ middle @ right, with middle None standing for the identity."""
    if middle is None:
        return left.conj().T @ right
    return left.conj().T @ (middle @ right)


def _with_coefficients(multi_indices: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Pair every multi-index with its coefficient, so the two cannot fall out of step."""
    return zip(_multinomial_coefficient(multi_indices), multi_indices)


def group_eigenspace(eigenvalues: np.ndarray, *, tol: float = 1e-5) -> np.ndarray:
    """Group eigenvalues according to degeneracy."""
    eigenvalues = np.asarray(eigenvalues).flatten()
    if eigenvalues.size == 0:
        return np.array([], dtype=int)

    sort_idx = np.argsort(eigenvalues)
    sorted_ev = eigenvalues[sort_idx]

    diffs = np.diff(sorted_ev)
    new_group_markers = diffs >= tol

    sorted_groups = np.concatenate(([0], np.cumsum(new_group_markers)), dtype=int)

    groups = np.empty_like(sorted_groups)
    groups[sort_idx] = sorted_groups

    return groups
