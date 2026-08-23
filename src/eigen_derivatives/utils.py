import itertools
import math

import numpy as np
import scipy.sparse as sp


def _get_numeric_backend(is_sparse: bool):
    """
    Returns the appropriate function bindings and matrix factories
    based on whether a sparse or dense backend is requested.
    """
    if is_sparse:
        return (
            sp.csc_matrix,
            sp.bmat,
            lambda system_matrix, rhs: sp.linalg.spsolve(system_matrix.tocsc(), rhs),
            lambda rows, cols: sp.csc_matrix((rows, cols)),
            lambda dim: sp.eye(dim, format='csc')
        )


    return (
        np.asarray,
        np.block,
        np.linalg.solve,
        lambda rows, cols: np.zeros((rows, cols)),
        lambda dim: np.eye(dim)
    )


def _multiindex_total_order(total_order: int, length: int) -> np.ndarray:
    """All multi-indices of a specific length and order."""
    if length <= 0:
        return np.empty((0, 0), dtype=int)
    if total_order < 0:
        return np.empty((0, length), dtype=int)
    if length == 1:
        return np.array([[total_order]], dtype=int)

    pools = [range(total_order + 1)] * length
    indices = [
        p for p in itertools.product(*pools)
        if sum(p) == total_order
    ]
    return np.array(indices, dtype=int)


def _multinomial_coefficient(list_of_multi_indices) -> list[int]:
    """Calculates multinomial coefficients."""
    multi_indices = np.asarray(list_of_multi_indices, dtype=int)
    if multi_indices.ndim == 1:
        multi_indices = multi_indices.reshape(1, -1)

    coefficients = []
    for row in multi_indices:
        n = int(np.sum(row))
        denominator = math.prod(math.factorial(int(x)) for x in row)
        coefficients.append(math.factorial(n) // denominator)
    return coefficients


def group_eigenspace(eigenvalues, tol: float = 1e-5) -> np.ndarray:
    """Groups eigenvalues according to degeneracy."""
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