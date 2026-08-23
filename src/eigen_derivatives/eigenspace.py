import numpy as np
import scipy.sparse as sp

from eigen_derivatives.derivatives_list import DerivativesList
from eigen_derivatives.utils import _multiindex_total_order, _multinomial_coefficient, _get_numeric_backend


def eigenpair_derivatives(
        unperturbed_eigenvalue: float,
        unperturbed_eigenfunctions: np.ndarray,
        stiffness_matrix: DerivativesList,
        mass_matrix: DerivativesList | None = None
) -> tuple[DerivativesList, DerivativesList]:
    """Derivatives of eigenpairs (with respect to the eigenspace)."""
    dof, multiplicity = unperturbed_eigenfunctions.shape
    num_orders = len(stiffness_matrix)

    is_sparse_type = sp.issparse(stiffness_matrix[0])
    to_matrix, block_func, solve_func, make_zero, make_eye = _get_numeric_backend(is_sparse_type)

    zero_block = make_zero(multiplicity, multiplicity)

    if mass_matrix is None:
        mass_matrix = DerivativesList([make_eye(dof)])
    has_mass_matrix_derivatives = len(mass_matrix) > 1

    eigenvalue_derivatives: list = (
            [unperturbed_eigenvalue * np.eye(multiplicity)]
            + [np.zeros((multiplicity, multiplicity)) for _ in range(num_orders - 1)]
    )
    eigenvector_derivatives: list = (
            [unperturbed_eigenfunctions]
            + [np.zeros((dof, multiplicity)) for _ in range(num_orders - 1)]
    )

    northwest_tile = to_matrix(stiffness_matrix[0] - mass_matrix[0] * unperturbed_eigenvalue)
    northeast_tile = to_matrix(-mass_matrix[0] @ eigenvector_derivatives[0])

    system_matrix = block_func([
        [northwest_tile, northeast_tile],
        [northeast_tile.T, zero_block]
    ])

    for k in range(1, num_orders):
        if not has_mass_matrix_derivatives:
            diagonal_term = np.zeros((multiplicity, multiplicity))
        else:
            diagonal_term_vec = np.zeros(multiplicity)
            multi_indices = _multiindex_total_order(k, 3)
            multi_indices = multi_indices[(multi_indices[:, 0] < k) & (multi_indices[:, 2] < k)]
            mnc = _multinomial_coefficient(multi_indices)

            for coeff, ind in zip(mnc, multi_indices):
                prod = eigenvector_derivatives[ind[0]].T @ (mass_matrix[ind[1]] @ eigenvector_derivatives[ind[2]])
                val_vec = np.ravel(prod.diagonal())
                diagonal_term_vec += (coeff / 2.0) * val_vec

            diagonal_term = np.diag(diagonal_term_vec)

        rhs_n = np.zeros((dof, multiplicity))

        multi_indices = _multiindex_total_order(k, 2)
        multi_indices = multi_indices[multi_indices[:, 1] < k]
        mnc = _multinomial_coefficient(multi_indices)
        for coeff_A, ind in zip(mnc, multi_indices):
            rhs_n += -coeff_A * (stiffness_matrix[ind[0]] @ eigenvector_derivatives[ind[1]])

        multi_indices = _multiindex_total_order(k, 3)
        multi_indices = multi_indices[(multi_indices[:, 1] < k) & (multi_indices[:, 2] < k)]
        if not has_mass_matrix_derivatives:
            multi_indices = multi_indices[multi_indices[:, 0] == 0]
        mnc = _multinomial_coefficient(multi_indices)
        for coeff, ind in zip(mnc, multi_indices):
            rhs_n += coeff * (mass_matrix[ind[0]] @ eigenvector_derivatives[ind[1]] @ eigenvalue_derivatives[ind[2]])

        rhs = np.vstack([rhs_n, diagonal_term])

        sol = solve_func(system_matrix, rhs)

        eigenvector_derivatives[k] = sol[:dof]
        eigenvalue_derivatives[k] = sol[dof:]

    return DerivativesList(eigenvalue_derivatives), DerivativesList(eigenvector_derivatives)
