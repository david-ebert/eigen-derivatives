import warnings

import numpy as np

from eigen_derivatives._types import Matrix
from eigen_derivatives.backend import Backend, DenseBackend
from eigen_derivatives.derivative_series import DerivativeSeries
from eigen_derivatives.utils import (
    _bilinear_form, _multiindex_total_order, _multinomial_coefficient, _validate_mass_series,
    _with_coefficients, group_eigenspace
)


def polarize(
        eigenvalue_ds: DerivativeSeries | list[Matrix | None], *,
        tol: float = 1e-5,
        k: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return the initial polarization and the order at which each pair of eigenvalues separates.

    The order matrix is float. Pairs that the supplied derivatives do not separate are
    marked with infinity, so that np.isinf finds them.
    """
    n = eigenvalue_ds[-1].shape[0]
    num_orders = len(eigenvalue_ds)

    polarization_order = np.ones((n, n), dtype=np.float64) * k
    eigh_res = np.linalg.eigh(eigenvalue_ds[k])
    split_eigenvalues, init_polarization = eigh_res.eigenvalues, eigh_res.eigenvectors
    group = group_eigenspace(split_eigenvalues, tol=tol)
    polarization_adjustment = np.eye(n, dtype=float)

    for i in range(np.max(group) + 1):
        mask = (group == i)
        if np.sum(mask) > 1:
            mask_2d = np.ix_(mask, mask)
            if k < num_orders - 1:
                derivatives_submatrix = [None] * num_orders
                init_polarization_col = init_polarization[:, mask]

                for j in range(k + 1, num_orders):
                    derivatives_submatrix[j] = _bilinear_form(
                        init_polarization_col, init_polarization_col, middle=eigenvalue_ds[j]
                    )

                polarization_submatrix, polarization_order_sub = polarize(derivatives_submatrix, tol=tol, k=k + 1)

                polarization_adjustment[mask_2d] = polarization_submatrix
                polarization_order[mask_2d] = polarization_order_sub
            else:
                polarization_order[mask_2d] = np.inf

    init_polarization = init_polarization @ polarization_adjustment

    if k == 1 and np.any(np.isinf(polarization_order)):
        warnings.warn(
            "No more derivatives of eigenvalues with respect to eigenspace available. "
            "Stopped and assumed that eigenvalues are identical.",
            UserWarning
        )

    return init_polarization, polarization_order


def polarization_derivatives(
        eigenvalue_ds: DerivativeSeries,
        eigenvector_ds: DerivativeSeries,
        init_polarization: np.ndarray,
        polarization_order: np.ndarray,
        mass_mat_ds: DerivativeSeries | None = None,
        *,
        backend: Backend | None = None
) -> tuple[DerivativeSeries, DerivativeSeries]:
    """Return the polarization matrix derivatives and the polarized eigenvalue derivatives.

    Entries of the polarization derivatives that the supplied eigenvalue derivatives do not
    determine are returned as NaN.
    """

    def _coefficient_builder(i: int, j: int, k_p: int, k_ij: int) -> float:
        accumulator = 0.0
        multi_indices = _multiindex_total_order(k_p + k_ij, 2)
        multi_indices = multi_indices[multi_indices[:, 0] >= k_ij]

        for coeff, ind in _with_coefficients(multi_indices):
            mat_diff = eigenvalue_ds[ind[0]] - polarized_eigenvalue_derivatives[ind[0]][i] * np.eye(multiplicity)
            accumulator += coeff * _bilinear_form(
                polarization_matrix_derivatives[0][:, j],
                polarization_matrix_derivatives[ind[1]][:, i],
                middle=mat_diff,
            )

        denom_coeff = np.asarray(_multinomial_coefficient(np.array([k_ij, k_p]))).item()
        denominator = (
                denom_coeff * (polarized_eigenvalue_derivatives[k_ij][j] - polarized_eigenvalue_derivatives[k_ij][i])
        )
        return accumulator / denominator

    multiplicity = init_polarization.shape[0]
    num_orders = len(eigenvalue_ds)

    # An omitted mass matrix stays omitted; building the identity would be dof by dof.
    has_mass_matrix = mass_mat_ds is not None
    if has_mass_matrix:
        _validate_mass_series(mass_mat_ds, num_orders)
    has_mass_matrix_derivatives = has_mass_matrix and len(mass_mat_ds) > 1

    polarization_matrix_derivatives = (
            [init_polarization] + [np.zeros((multiplicity, multiplicity)) for _ in range(num_orders - 1)]
    )
    polarized_eigenvalue_derivatives = (
            [np.diag(eigenvalue_ds[0])] + [np.zeros(multiplicity) for _ in range(num_orders - 1)]
    )
    determined = np.zeros((num_orders - 1, multiplicity), dtype=bool)

    # not factorized: the system is multiplicity by multiplicity
    backend = DenseBackend() if backend is None else backend
    lhs = -init_polarization.conj().T

    for k_eigval in range(1, num_orders):
        for i in range(multiplicity):
            if k_eigval <= polarization_order[i, i]:
                polarized_eigenvalue_derivatives[k_eigval][i] = _bilinear_form(
                    init_polarization[:, i], init_polarization[:, i],
                    middle=eigenvalue_ds[k_eigval],
                )
                continue

            rhs = np.zeros(multiplicity)
            k_ii = int(polarization_order[i, i])
            k_p = k_eigval - k_ii

            for j in range(multiplicity):
                k_ij = int(polarization_order[i, j])
                if k_ij < k_ii:
                    rhs[j] = _coefficient_builder(i, j, k_p, k_ij)

            polarization_matrix_derivatives[k_p][:, i] = backend.solve(lhs, rhs)

            rhs_vec: np.ndarray = eigenvalue_ds[k_eigval] @ init_polarization[:, i]
            multi_indices = _multiindex_total_order(k_eigval, 2)
            multi_indices = multi_indices[
                (multi_indices[:, 0] >= k_ii) & (multi_indices[:, 0] < k_eigval)
                ]
            for coeff, ind in _with_coefficients(multi_indices):
                polarization_column = polarization_matrix_derivatives[ind[1]][:, i]
                rhs_vec += coeff * (
                        eigenvalue_ds[ind[0]] @ polarization_column
                        - polarization_column * polarized_eigenvalue_derivatives[ind[0]][i]
                )

            polarized_eigenvalue_derivatives[k_eigval][i] = _bilinear_form(init_polarization[:, i], rhs_vec)

            rhs = np.zeros(multiplicity)
            for j in range(multiplicity):
                k_ij = int(polarization_order[i, j])
                if k_ij == k_ii:
                    if i != j:
                        rhs[j] = _coefficient_builder(i, j, k_p, k_ij)
                    else:
                        multi_indices = _multiindex_total_order(k_p, 5)
                        mask_norm = (multi_indices[:, 0] < k_p) & (multi_indices[:, 4] < k_p)
                        if not has_mass_matrix_derivatives:
                            mask_norm &= (multi_indices[:, 2] == 0)
                        multi_indices = multi_indices[mask_norm]
                        for coeff, ind in _with_coefficients(multi_indices):
                            right = (
                                    eigenvector_ds[ind[3]]
                                    @ polarization_matrix_derivatives[ind[4]][:, i]
                            )
                            inner = _bilinear_form(
                                eigenvector_ds[ind[1]], right,
                                middle=mass_mat_ds[ind[2]] if has_mass_matrix else None,
                            )
                            rhs[j] += (coeff / 2.0) * _bilinear_form(
                                polarization_matrix_derivatives[ind[0]][:, i], inner
                            )

            polarization_matrix_derivatives[k_p][:, i] = backend.solve(lhs, rhs)
            determined[k_p - 1, i] = True

    for order in range(1, num_orders):
        polarization_matrix_derivatives[order][:, ~determined[order - 1]] = np.nan

    return (
        DerivativeSeries(tuple(polarization_matrix_derivatives)),
        DerivativeSeries(tuple(polarized_eigenvalue_derivatives)),
    )


def polarized_eigenvectors(
        eigenvector_ds: DerivativeSeries,
        polarization_ds: DerivativeSeries
) -> DerivativeSeries:
    """Return the polarized eigenvector derivatives."""
    num_orders = len(polarization_ds)
    if num_orders == 1:
        return DerivativeSeries(tuple(eigenvector @ polarization_ds[0] for eigenvector in eigenvector_ds))

    eigenvector_shape = eigenvector_ds[0].shape
    polarized_list = []

    for k in range(0, num_orders):
        multi_indices = _multiindex_total_order(k, 2)
        polarized = np.zeros(eigenvector_shape)

        for coeff, ind in _with_coefficients(multi_indices):
            polarized += coeff * (eigenvector_ds[ind[0]] @ polarization_ds[ind[1]])
        polarized_list.append(polarized)

    return DerivativeSeries(tuple(polarized_list))
