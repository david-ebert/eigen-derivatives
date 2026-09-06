import tracemalloc

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from eigen_derivatives import (
    DerivativeSeries,
    eigenpair_derivatives,
    polarization_derivatives,
    polarize,
    polarized_eigenvectors,
)

EVALUATION = 2.0 * np.eye(2)
FIRST_DERIVATIVE = np.array([[1.0, -1.0], [-1.0, -1.0]])
SECOND_DERIVATIVE = np.zeros((2, 2))
EXACT_EIGENVALUES = np.array([2.0 - np.sqrt(2.0), 2.0 + np.sqrt(2.0)])


def _stiffness(sparse: bool) -> DerivativeSeries:
    if sparse:
        return DerivativeSeries(
            (sp.csc_matrix(EVALUATION), sp.csc_matrix(FIRST_DERIVATIVE), sp.csc_matrix(SECOND_DERIVATIVE))
        )
    return DerivativeSeries((EVALUATION, FIRST_DERIVATIVE, SECOND_DERIVATIVE))


def _run_workflow(sparse: bool):
    stiffness_ds = _stiffness(sparse)
    unperturbed = np.linalg.eigh(EVALUATION)
    eigenvalue_ds, eigenvector_ds = eigenpair_derivatives(
        unperturbed.eigenvalues[0], unperturbed.eigenvectors, stiffness_ds
    )
    initial_polarization, polarization_order = polarize(eigenvalue_ds)
    polarization_ds, polarized_eigenvalue_ds = polarization_derivatives(
        eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order
    )
    eigenvector_ds = polarized_eigenvectors(eigenvector_ds, polarization_ds)
    return polarized_eigenvalue_ds, eigenvector_ds, polarization_order, polarization_ds


class TestPolarizationWorkflow:
    @pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
    def test_taylor_reproduces_the_exact_eigenvalues(self, sparse):
        polarized_eigenvalue_ds, _, _, _ = _run_workflow(sparse)
        approximation = np.sort(np.ravel(polarized_eigenvalue_ds.evaluate_taylor(1.0)))
        assert np.allclose(approximation, EXACT_EIGENVALUES)

    @pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
    def test_polarizing_order_is_one_for_a_first_order_split(self, sparse):
        _, _, polarization_order, _ = _run_workflow(sparse)
        assert np.all(polarization_order == 1)

    @pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
    def test_polarized_eigenvectors_keep_the_series_length_and_shape(self, sparse):
        _, eigenvector_ds, _, _ = _run_workflow(sparse)
        assert len(eigenvector_ds) == 3
        assert eigenvector_ds.shape == (2, 2)

    @pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
    def test_output_is_always_dense(self, sparse):
        polarized_eigenvalue_ds, eigenvector_ds, _, _ = _run_workflow(sparse)
        assert all(isinstance(element, np.ndarray) for element in polarized_eigenvalue_ds)
        assert all(isinstance(element, np.ndarray) for element in eigenvector_ds)

    def test_dense_and_sparse_agree(self):
        dense_eigenvalue_ds, dense_eigenvector_ds, _, _ = _run_workflow(sparse=False)
        sparse_eigenvalue_ds, sparse_eigenvector_ds, _, _ = _run_workflow(sparse=True)
        assert all(
            np.allclose(a, b) for a, b in zip(dense_eigenvalue_ds, sparse_eigenvalue_ds, strict=True)
        )
        assert all(
            np.allclose(a, b, equal_nan=True)
            for a, b in zip(dense_eigenvector_ds, sparse_eigenvector_ds, strict=True)
        )


class TestUndeterminedEntries:
    @pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
    def test_polarized_eigenvalues_stay_finite(self, sparse):
        polarized_eigenvalue_ds, _, _, _ = _run_workflow(sparse)
        assert all(np.isfinite(element).all() for element in polarized_eigenvalue_ds)

    @pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
    def test_determined_polarization_orders_stay_finite(self, sparse):
        _, _, _, polarization_ds = _run_workflow(sparse)
        assert np.isfinite(polarization_ds[0]).all()
        assert np.isfinite(polarization_ds[1]).all()

    @pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
    def test_the_highest_order_is_not_determined(self, sparse):
        _, _, _, polarization_ds = _run_workflow(sparse)
        assert np.isnan(polarization_ds[len(polarization_ds) - 1]).all()

    @pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
    def test_the_order_matrix_is_float_in_the_ordinary_case(self, sparse):
        _, _, polarization_order, _ = _run_workflow(sparse)
        assert polarization_order.dtype == np.float64
        assert np.isfinite(polarization_order).all()

    def test_the_order_matrix_keeps_its_dtype_when_the_derivatives_run_out(self):
        eigenvalue_ds = DerivativeSeries((np.zeros((2, 2)), np.eye(2)))
        with pytest.warns(UserWarning, match="No more derivatives"):
            _, polarization_order = polarize(eigenvalue_ds)
        assert polarization_order.dtype == np.float64
        assert np.isinf(polarization_order).all()

    def test_exhaustion_inside_the_recursion_marks_only_the_affected_block(self):
        eigenvalue_ds = DerivativeSeries((np.zeros((3, 3)), np.diag([1.0, 1.0, 2.0]), np.eye(3)))
        eigenvector_ds = DerivativeSeries((np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))))
        with pytest.warns(UserWarning, match="No more derivatives"):
            initial_polarization, polarization_order = polarize(eigenvalue_ds)

        assert polarization_order.dtype == np.float64
        expected = np.array([[True, True, False], [True, True, False], [False, False, False]])
        assert np.array_equal(np.isinf(polarization_order), expected)

        polarization_ds, _ = polarization_derivatives(
            eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order
        )
        not_separated = np.isinf(np.diag(polarization_order))
        for order in range(1, len(polarization_ds)):
            assert np.isnan(polarization_ds[order][:, not_separated]).all()

    def test_exhaustion_keeps_the_orders_between_sibling_blocks(self):
        eigenvalue_ds = DerivativeSeries((np.zeros((4, 4)), np.diag([1.0, 1.0, 2.0, 2.0])))
        with pytest.warns(UserWarning, match="No more derivatives"):
            _, polarization_order = polarize(eigenvalue_ds)

        undetermined = np.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, True],
                [False, False, True, True],
            ]
        )
        assert np.array_equal(np.isinf(polarization_order), undetermined)
        assert np.all(polarization_order[~undetermined] == 1.0)

    def test_everything_is_nan_when_the_derivatives_run_out(self):
        eigenvalue_ds = DerivativeSeries((np.zeros((2, 2)), np.eye(2)))
        eigenvector_ds = DerivativeSeries((np.eye(2), np.zeros((2, 2))))
        with pytest.warns(UserWarning, match="No more derivatives"):
            initial_polarization, polarization_order = polarize(eigenvalue_ds)
        polarization_ds, _ = polarization_derivatives(
            eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order
        )
        assert np.isnan(polarization_ds[1]).all()


def _rectangular_case():
    """A degenerate pair inside a larger space, so dof exceeds the multiplicity."""
    rng = np.random.default_rng(3)
    rotation, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    spectra = [
        np.diag([1.0, 1.0, 3.0, 4.0, 6.0]),
        np.diag([1.0, -1.0, 1.0, 2.0, 0.5]),
        np.diag([0.7, 0.2, 1.0, 0.0, 0.3]),
        np.diag([1.3, -0.4, 0.2, 0.1, 0.0]),
    ]
    coupling = rng.standard_normal((5, 5))
    coupling = 0.5 * (coupling + coupling.T)
    stiffness = [rotation @ spectra[0] @ rotation.T] + [
        rotation @ (spectrum + 0.3 * coupling) @ rotation.T for spectrum in spectra[1:]
    ]
    return DerivativeSeries(tuple(stiffness)), rotation[:, :2]


def _rectangular_generalized_case():
    """The same, with a non-trivial mass matrix and mass-orthonormal eigenvectors."""
    rng = np.random.default_rng(11)
    dof = 5
    factor = rng.standard_normal((dof, dof))
    mass = factor @ factor.T + dof * np.eye(dof)
    cholesky_factor = np.linalg.cholesky(mass)
    orthogonal, _ = np.linalg.qr(rng.standard_normal((dof, dof)))
    mass_orthonormal = sla.solve_triangular(cholesky_factor.T, orthogonal, lower=False)

    def symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    stiffness_evaluation = (
            mass @ mass_orthonormal @ np.diag([1.0, 1.0, 3.0, 4.0, 6.0]) @ mass_orthonormal.T @ mass
    )
    stiffness = (stiffness_evaluation,) + tuple(
        symmetrize(rng.standard_normal((dof, dof))) for _ in range(3)
    )
    mass_elements = [mass] + [0.05 * symmetrize(rng.standard_normal((dof, dof))) for _ in range(3)]
    return DerivativeSeries(stiffness), mass_elements, mass_orthonormal[:, :2]


def _rectangular_workflow():
    stiffness_ds, initial_eigenvectors = _rectangular_case()
    eigenvalue_ds, eigenvector_ds = eigenpair_derivatives(1.0, initial_eigenvectors, stiffness_ds)
    initial_polarization, polarization_order = polarize(eigenvalue_ds)
    return eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order


class TestMoreDofsThanMultiplicity:
    """The eigenspace is a strict subspace, so the eigenvector blocks are rectangular."""

    def test_the_workflow_runs_and_the_series_keep_their_shapes(self):
        eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order = _rectangular_workflow()
        polarization_ds, _ = polarization_derivatives(
            eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order
        )
        assert eigenvector_ds.shape == (5, 2)
        assert polarization_ds.shape == (2, 2)
        assert np.isfinite(polarization_ds[1]).all()

    def test_polarized_eigenvectors_stay_orthonormal_to_the_truncation_order(self):
        eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order = _rectangular_workflow()
        polarization_ds, _ = polarization_derivatives(
            eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order
        )
        polarized_ds = polarized_eigenvectors(eigenvector_ds, polarization_ds)

        residuals = []
        for step in (0.1, 0.05, 0.025):
            approximation = polarized_ds.truncate(2).evaluate_taylor(step)
            residuals.append(np.linalg.norm(approximation.T @ approximation - np.eye(2)))

        # an order 2 truncation leaves an O(step^3) defect, so halving step divides by eight
        for coarse, fine in zip(residuals[:-1], residuals[1:]):
            assert 6.0 < coarse / fine < 10.0

    def test_an_omitted_mass_matrix_matches_an_explicit_identity(self):
        eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order = _rectangular_workflow()
        without_mass, eigenvalues_without = polarization_derivatives(
            eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order
        )
        identity_ds = DerivativeSeries((np.eye(eigenvector_ds.shape[0]),))
        with_identity, eigenvalues_with = polarization_derivatives(
            eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order, identity_ds
        )

        for omitted, explicit in zip(without_mass, with_identity, strict=True):
            assert np.allclose(omitted, explicit, equal_nan=True)
        for omitted, explicit in zip(eigenvalues_without, eigenvalues_with, strict=True):
            assert np.allclose(omitted, explicit, equal_nan=True)

    def test_an_omitted_mass_matrix_does_not_allocate_a_dense_identity(self):
        dof = 1500
        diagonal = np.concatenate(([1.0, 1.0], np.arange(3.0, dof + 1.0)))
        stiffness_ds = DerivativeSeries((
            sp.diags(diagonal, format="csc"),
            sp.diags(np.concatenate(([1.0, -1.0], np.zeros(dof - 2))), format="csc"),
            sp.diags(np.concatenate(([0.5, 0.2], np.zeros(dof - 2))), format="csc"),
        ))
        initial_eigenvectors = np.zeros((dof, 2))
        initial_eigenvectors[0, 0] = 1.0
        initial_eigenvectors[1, 1] = 1.0

        eigenvalue_ds, eigenvector_ds = eigenpair_derivatives(1.0, initial_eigenvectors, stiffness_ds)
        initial_polarization, polarization_order = polarize(eigenvalue_ds)

        tracemalloc.start()
        polarization_derivatives(eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < dof * dof * 8 / 10

    def test_a_sparse_mass_matrix_matches_its_dense_counterpart(self):
        stiffness_ds, mass_elements, initial_eigenvectors = _rectangular_generalized_case()

        def run(mass_elements):
            mass_ds = DerivativeSeries(tuple(mass_elements))
            eigenvalue_ds, eigenvector_ds = eigenpair_derivatives(
                1.0, initial_eigenvectors, stiffness_ds, mass_ds
            )
            initial_polarization, polarization_order = polarize(eigenvalue_ds)
            return polarization_derivatives(
                eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order, mass_ds
            )

        dense_ds, dense_eigenvalues = run(mass_elements)
        sparse_ds, sparse_eigenvalues = run([sp.csc_matrix(mass) for mass in mass_elements])

        for dense, sparse in zip(dense_ds, sparse_ds, strict=True):
            assert np.allclose(dense, sparse, equal_nan=True)
        for dense, sparse in zip(dense_eigenvalues, sparse_eigenvalues, strict=True):
            assert np.allclose(dense, sparse, equal_nan=True)


class TestPolarizeWarnsWhenDerivativesRunOut:
    def test_identical_eigenvalues_warn(self):
        eigenvalue_ds = DerivativeSeries((np.zeros((2, 2)), np.eye(2)))
        with pytest.warns(UserWarning, match="No more derivatives"):
            polarize(eigenvalue_ds)
