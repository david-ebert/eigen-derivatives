import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp

from eigen_derivatives.derivatives_list import DerivativesList
from eigen_derivatives.utils import (
    _multiindex_total_order, _multinomial_coefficient, group_eigenspace, _get_numeric_backend
)


def polarize(
        eigenvalue_matrices: DerivativesList | list[Any],
        tol: float = 1e-5,
        k: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Initial Polarization P0 and decision matrix polarizing_order."""
    n = eigenvalue_matrices[-1].shape[0]
    d = len(eigenvalue_matrices)

    polarizing_order = np.ones((n, n), dtype=int) * k
    evals, initial_polarization = np.linalg.eigh(eigenvalue_matrices[k])
    group = group_eigenspace(evals, tol)
    polarization_adjustment = np.eye(n, dtype=float)

    for i in range(np.max(group) + 1):
        mask = (group == i)
        if np.sum(mask) > 1:
            if k < d - 1:
                derivatives_submatrix = [None] * d
                initial_polarization_col = initial_polarization[:, mask]

                for j in range(k + 1, d):
                    derivatives_submatrix[j] = (
                            initial_polarization_col.T
                            @ eigenvalue_matrices[j]
                            @ initial_polarization_col
                    )

                polarization_submatrix, polarizing_order_sub = polarize(
                    derivatives_submatrix, tol, k + 1
                )

                mask_2d = np.ix_(mask, mask)
                polarization_adjustment[mask_2d] = polarization_submatrix
                polarizing_order[mask_2d] = polarizing_order_sub
            else:
                polarizing_order = np.full((n, n), np.inf)
                break

    initial_polarization = initial_polarization @ polarization_adjustment

    if k == 1 and np.any(np.isinf(polarizing_order)):
        warnings.warn(
            "No more derivatives of eigenvalues with respect to eigenspace available. "
            "Stopped and assumed that eigenvalues are identical.",
            UserWarning
        )

    return initial_polarization, polarizing_order


def polarization_derivatives(
        eigenvalue_derivatives: DerivativesList,
        eigenfunction_derivatives: DerivativesList,
        initial_polarization: np.ndarray,
        polarization_order: np.ndarray,
        mass_matrix: DerivativesList | None = None
) -> tuple[DerivativesList, DerivativesList]:
    """Calculates available polarized derivatives of dl and corresponding dQ natively."""

    def _coefficient_builder(i: int, j: int, k_Q: int, kij: int) -> float:
        acc = 0.0
        multi_indices = _multiindex_total_order(k_Q + kij, 2)
        multi_indices = multi_indices[multi_indices[:, 0] >= kij]

        mnc = _multinomial_coefficient(multi_indices)
        for coeff, ind in zip(mnc, multi_indices):
            mat_diff = (
                    eigenvalue_derivatives[ind[0]]
                    - polarized_eigenvalue_derivatives[ind[0]][i]
                    * np.eye(multiplicity)
            )
            acc += (
                    coeff
                    * polarization_derivatives[0][:, j].T
                    @ mat_diff
                    @ polarization_derivatives[ind[1]][:, i]
            )

        denom_coeff = np.asarray(_multinomial_coefficient(np.array([kij, k_Q]))).item()
        denominator = (
                denom_coeff * ( polarized_eigenvalue_derivatives[kij][j] - polarized_eigenvalue_derivatives[kij][i] )
        )
        return acc / denominator

    multiplicity = initial_polarization.shape[0]
    d = len(eigenvalue_derivatives)
    dof = eigenfunction_derivatives.shape[0]

    is_sparse_type = sp.issparse(eigenfunction_derivatives[0])
    to_matrix, _, solve_func, _, make_eye = _get_numeric_backend(
        is_sparse_type
    )

    if mass_matrix is None:
        mass_matrix = DerivativesList([make_eye(dof)])
    has_mass_matrix_derivatives = len(mass_matrix) > 1

    polarization_derivatives = [initial_polarization] + [
        np.zeros((multiplicity, multiplicity)) for _ in range(d - 1)
    ]
    polarized_eigenvalue_derivatives = [
                                           np.diag(eigenvalue_derivatives[0])
                                       ] + [np.zeros(multiplicity) for _ in range(d - 1)]

    lhs = -initial_polarization.T

    for k_l in range(1, d):
        for i in range(multiplicity):
            if k_l <= polarization_order[i, i]:
                polarized_eigenvalue_derivatives[k_l][i] = (
                        initial_polarization[:, i].T
                        @ eigenvalue_derivatives[k_l]
                        @ initial_polarization[:, i]
                )
                continue

            rhs = np.zeros(multiplicity)
            kii = int(polarization_order[i, i])
            k_P = k_l - kii

            for j in range(multiplicity):
                kij = int(polarization_order[i, j])
                if kij < kii:
                    rhs[j] = _coefficient_builder(i, j, k_P, kij)

            polarization_derivatives[k_P][:, i] = solve_func(lhs, rhs)

            rhs_vec = eigenvalue_derivatives[k_l] @ initial_polarization[:, i]
            multi_indices = _multiindex_total_order(k_l, 2)
            multi_indices = multi_indices[
                (multi_indices[:, 0] >= kii) & (multi_indices[:, 0] < k_l)
                ]
            mnc = _multinomial_coefficient(multi_indices)
            for coeff, ind in zip(mnc, multi_indices):
                pol_col = polarization_derivatives[ind[1]][:, i]
                rhs_vec += coeff * (
                        eigenvalue_derivatives[ind[0]] @ pol_col
                        - pol_col * polarized_eigenvalue_derivatives[ind[0]][i]
                )

            polarized_eigenvalue_derivatives[k_l][i] = (
                    initial_polarization[:, i].T @ rhs_vec
            )

            rhs = np.zeros(multiplicity)
            for j in range(multiplicity):
                kij = int(polarization_order[i, j])
                if kij == kii:
                    if i != j:
                        rhs[j] = _coefficient_builder(i, j, k_P, kij)
                    else:
                        multi_indices = _multiindex_total_order(k_P, 5)
                        mask_norm = (multi_indices[:, 0] < k_P) & (
                                multi_indices[:, 4] < k_P
                        )
                        if not has_mass_matrix_derivatives:
                            mask_norm &= (multi_indices[:, 2] == 0)
                        multi_indices = multi_indices[mask_norm]
                        mnc = _multinomial_coefficient(multi_indices)

                        for coeff, ind in zip(mnc, multi_indices):
                            val_scalar = polarization_derivatives[ind[0]][:, i].T @ (
                                    eigenfunction_derivatives[ind[1]] @ (
                                    mass_matrix[ind[2]] @ (
                                    eigenfunction_derivatives[ind[3]]
                                    @ polarization_derivatives[ind[4]][:, i])))

                            rhs[j] += (coeff / 2.0) * val_scalar

            polarization_derivatives[k_P][:, i] = solve_func(lhs, rhs)

    return (
        DerivativesList(polarization_derivatives),
        DerivativesList(polarized_eigenvalue_derivatives)
    )


def polarized_eigenvectors(
        eigenvector_derivatives: DerivativesList,
        polarization_derivatives: DerivativesList
) -> DerivativesList:
    """Calculates polarized eigenfunction derivatives."""
    d = len(polarization_derivatives)
    if d == 1:
        return DerivativesList([
            u_item @ polarization_derivatives[0]
            for u_item in eigenvector_derivatives
        ])

    size_u = eigenvector_derivatives[0].shape
    du_pol_list = []

    for k in range(0, d):
        multi_indices = _multiindex_total_order(k, 2)
        mnc = _multinomial_coefficient(multi_indices)
        du_pol = np.zeros(size_u)

        for coeff, ind in zip(mnc, multi_indices):
            if ind[0] >= len(eigenvector_derivatives):
                continue

            du_pol += coeff * (
                    eigenvector_derivatives[ind[0]]
                    @ polarization_derivatives[ind[1]]
            )
        du_pol_list.append(du_pol)

    return DerivativesList(du_pol_list)
