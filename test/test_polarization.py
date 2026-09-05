import numpy as np
import pytest
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
    @pytest.mark.parametrize("sparse", [False, True])
    def test_taylor_reproduces_the_exact_eigenvalues(self, sparse):
        polarized_eigenvalue_ds, _, _, _ = _run_workflow(sparse)
        approximation = np.sort(np.ravel(polarized_eigenvalue_ds.evaluate_taylor(1.0)))
        assert np.allclose(approximation, EXACT_EIGENVALUES)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_polarizing_order_is_one_for_a_first_order_split(self, sparse):
        _, _, polarization_order, _ = _run_workflow(sparse)
        assert np.all(polarization_order == 1)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_polarized_eigenvectors_keep_the_series_length_and_shape(self, sparse):
        _, eigenvector_ds, _, _ = _run_workflow(sparse)
        assert len(eigenvector_ds) == 3
        assert eigenvector_ds.shape == (2, 2)

    @pytest.mark.parametrize("sparse", [False, True])
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
    @pytest.mark.parametrize("sparse", [False, True])
    def test_polarized_eigenvalues_stay_finite(self, sparse):
        polarized_eigenvalue_ds, _, _, _ = _run_workflow(sparse)
        assert all(np.isfinite(element).all() for element in polarized_eigenvalue_ds)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_determined_polarization_orders_stay_finite(self, sparse):
        _, _, _, polarization_ds = _run_workflow(sparse)
        assert np.isfinite(polarization_ds[0]).all()
        assert np.isfinite(polarization_ds[1]).all()

    @pytest.mark.parametrize("sparse", [False, True])
    def test_the_highest_order_is_not_determined(self, sparse):
        _, _, _, polarization_ds = _run_workflow(sparse)
        assert np.isnan(polarization_ds[len(polarization_ds) - 1]).all()

    @pytest.mark.parametrize("sparse", [False, True])
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

    def test_everything_is_nan_when_the_derivatives_run_out(self):
        eigenvalue_ds = DerivativeSeries((np.zeros((2, 2)), np.eye(2)))
        eigenvector_ds = DerivativeSeries((np.eye(2), np.zeros((2, 2))))
        with pytest.warns(UserWarning, match="No more derivatives"):
            initial_polarization, polarization_order = polarize(eigenvalue_ds)
        polarization_ds, _ = polarization_derivatives(
            eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order
        )
        assert np.isnan(polarization_ds[1]).all()


class TestPolarizeWarnsWhenDerivativesRunOut:
    def test_identical_eigenvalues_warn(self):
        eigenvalue_ds = DerivativeSeries((np.zeros((2, 2)), np.eye(2)))
        with pytest.warns(UserWarning, match="No more derivatives"):
            polarize(eigenvalue_ds)
