import numpy as np

from eigen_derivatives.backend import Backend, get_backend
from eigen_derivatives.derivative_series import DerivativeSeries
from eigen_derivatives.utils import (
    _bilinear_form, _multiindex_total_order, _validate_mass_series, _with_coefficients
)


def eigenpair_derivatives(
        eigenvalue: float,
        eigenvectors: np.ndarray,
        stiffness_mat_ds: DerivativeSeries,
        mass_mat_ds: DerivativeSeries | None = None,
        *,
        backend: Backend | None = None
) -> tuple[DerivativeSeries, DerivativeSeries]:
    """Return the eigenpair derivatives with respect to the eigenspace."""
    dof, multiplicity = eigenvectors.shape
    num_orders = len(stiffness_mat_ds)

    backend = get_backend(stiffness_mat_ds[0]) if backend is None else backend

    zero_block = backend.zeros(multiplicity, multiplicity)

    if mass_mat_ds is None:
        mass_mat_ds = DerivativeSeries((backend.eye(dof),))
    _validate_mass_series(mass_mat_ds, num_orders)
    has_mass_matrix_derivatives = len(mass_mat_ds) > 1

    eigenvalue_derivatives: list = (
            [eigenvalue * np.eye(multiplicity)]
            + [np.zeros((multiplicity, multiplicity)) for _ in range(num_orders - 1)]
    )
    eigenvector_derivatives: list = (
            [eigenvectors] + [np.zeros((dof, multiplicity)) for _ in range(num_orders - 1)]
    )

    northwest_tile = backend.as_matrix(stiffness_mat_ds[0] - mass_mat_ds[0] * eigenvalue)
    northeast_tile = backend.as_matrix(-mass_mat_ds[0] @ eigenvector_derivatives[0])

    system_matrix = backend.block([
        [northwest_tile, northeast_tile],
        [northeast_tile.conj().T, zero_block]
    ])

    # the system matrix is the same for every order, so it is factorized once
    solve_system = backend.factorize(system_matrix)

    for k in range(1, num_orders):
        diagonal = np.zeros(multiplicity)
        multi_indices = _multiindex_total_order(k, 3)
        multi_indices = multi_indices[(multi_indices[:, 0] < k) & (multi_indices[:, 2] < k)]
        if not has_mass_matrix_derivatives:
            multi_indices = multi_indices[multi_indices[:, 1] == 0]
        for coeff, ind in _with_coefficients(multi_indices):
            normalization = _bilinear_form(
                eigenvector_derivatives[ind[0]], eigenvector_derivatives[ind[2]],
                middle=mass_mat_ds[ind[1]],
            )
            diagonal += (coeff / 2.0) * np.ravel(normalization.diagonal())

        rhs_n = np.zeros((dof, multiplicity))

        multi_indices = _multiindex_total_order(k, 2)
        multi_indices = multi_indices[multi_indices[:, 1] < k]
        for coeff, ind in _with_coefficients(multi_indices):
            rhs_n += -coeff * (stiffness_mat_ds[ind[0]] @ eigenvector_derivatives[ind[1]])

        multi_indices = _multiindex_total_order(k, 3)
        multi_indices = multi_indices[(multi_indices[:, 1] < k) & (multi_indices[:, 2] < k)]
        if not has_mass_matrix_derivatives:
            multi_indices = multi_indices[multi_indices[:, 0] == 0]
        for coeff, ind in _with_coefficients(multi_indices):
            rhs_n += coeff * (mass_mat_ds[ind[0]] @ eigenvector_derivatives[ind[1]] @ eigenvalue_derivatives[ind[2]])

        rhs = np.vstack([rhs_n, np.diag(diagonal)])

        solution = solve_system(rhs)

        eigenvector_derivatives[k] = solution[:dof]
        eigenvalue_derivatives[k] = solution[dof:]

    return DerivativeSeries(tuple(eigenvalue_derivatives)), DerivativeSeries(tuple(eigenvector_derivatives))
