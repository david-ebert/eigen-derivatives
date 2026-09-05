import numpy as np
import scipy.sparse as sp

from eigen_derivatives.derivative_series import DerivativeSeries
from eigen_derivatives.utils import _multiindex_total_order, _multinomial_coefficient, _get_numeric_backend


def eigenpair_derivatives(
        eigenvalue: float,
        eigenvectors: np.ndarray,
        stiffness_mat_ds: DerivativeSeries,
        mass_mat_ds: DerivativeSeries | None = None
) -> tuple[DerivativeSeries, DerivativeSeries]:
    """Return the eigenpair derivatives with respect to the eigenspace."""
    dof, multiplicity = eigenvectors.shape
    num_orders = len(stiffness_mat_ds)

    is_sparse_type = sp.issparse(stiffness_mat_ds[0])
    to_matrix, block_func, solve_func, make_zero, make_eye = _get_numeric_backend(is_sparse_type)

    zero_block = make_zero(multiplicity, multiplicity)

    if mass_mat_ds is None:
        mass_mat_ds = DerivativeSeries((make_eye(dof),))
    has_mass_matrix_derivatives = len(mass_mat_ds) > 1

    eigenvalue_derivatives: list = (
            [eigenvalue * np.eye(multiplicity)]
            + [np.zeros((multiplicity, multiplicity)) for _ in range(num_orders - 1)]
    )
    eigenvector_derivatives: list = (
            [eigenvectors] + [np.zeros((dof, multiplicity)) for _ in range(num_orders - 1)]
    )

    northwest_tile = to_matrix(stiffness_mat_ds[0] - mass_mat_ds[0] * eigenvalue)
    northeast_tile = to_matrix(-mass_mat_ds[0] @ eigenvector_derivatives[0])

    system_matrix = block_func([
        [northwest_tile, northeast_tile],
        [northeast_tile.T, zero_block]
    ])

    for k in range(1, num_orders):
        diagonal_term_vec = np.zeros(multiplicity)
        multi_indices = _multiindex_total_order(k, 3)
        multi_indices = multi_indices[(multi_indices[:, 0] < k) & (multi_indices[:, 2] < k)]
        if not has_mass_matrix_derivatives:
            multi_indices = multi_indices[multi_indices[:, 1] == 0]
        coefficients = _multinomial_coefficient(multi_indices)

        for coeff, ind in zip(coefficients, multi_indices):
            gram_block = eigenvector_derivatives[ind[0]].T @ (mass_mat_ds[ind[1]] @ eigenvector_derivatives[ind[2]])
            diagonal_contribution = np.ravel(gram_block.diagonal())
            diagonal_term_vec += (coeff / 2.0) * diagonal_contribution

        diagonal_term = np.diag(diagonal_term_vec)

        rhs_n = np.zeros((dof, multiplicity))

        multi_indices = _multiindex_total_order(k, 2)
        multi_indices = multi_indices[multi_indices[:, 1] < k]
        coefficients = _multinomial_coefficient(multi_indices)
        for coeff_A, ind in zip(coefficients, multi_indices):
            rhs_n += -coeff_A * (stiffness_mat_ds[ind[0]] @ eigenvector_derivatives[ind[1]])

        multi_indices = _multiindex_total_order(k, 3)
        multi_indices = multi_indices[(multi_indices[:, 1] < k) & (multi_indices[:, 2] < k)]
        if not has_mass_matrix_derivatives:
            multi_indices = multi_indices[multi_indices[:, 0] == 0]
        coefficients = _multinomial_coefficient(multi_indices)
        for coeff, ind in zip(coefficients, multi_indices):
            rhs_n += coeff * (mass_mat_ds[ind[0]] @ eigenvector_derivatives[ind[1]] @ eigenvalue_derivatives[ind[2]])

        rhs = np.vstack([rhs_n, diagonal_term])

        solution = solve_func(system_matrix, rhs)

        eigenvector_derivatives[k] = solution[:dof]
        eigenvalue_derivatives[k] = solution[dof:]

    return DerivativeSeries(tuple(eigenvalue_derivatives)), DerivativeSeries(tuple(eigenvector_derivatives))
