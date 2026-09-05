import numpy as np
import pytest
import scipy.sparse as sp

from eigen_derivatives import DerivativeSeries, eigenpair_derivatives

EVALUATION = np.array([[1.0, 0.0], [0.0, 2.0]])
FIRST_DERIVATIVE = np.array([[1.0, -1.0], [-1.0, -1.0]])
SECOND_DERIVATIVE = np.zeros((2, 2))


def _stiffness(sparse: bool) -> DerivativeSeries:
    if sparse:
        return DerivativeSeries(
            (sp.csc_matrix(EVALUATION), sp.csc_matrix(FIRST_DERIVATIVE), sp.csc_matrix(SECOND_DERIVATIVE))
        )
    return DerivativeSeries((EVALUATION, FIRST_DERIVATIVE, SECOND_DERIVATIVE))


def _unperturbed_pair():
    result = np.linalg.eigh(EVALUATION)
    return result.eigenvalues[0], result.eigenvectors[:, 0:1]


class TestEigenpairDerivatives:
    def test_matches_first_order_perturbation_theory(self):
        eigenvalue, eigenvector = _unperturbed_pair()
        eigenvalue_ds, _ = eigenpair_derivatives(eigenvalue, eigenvector, _stiffness(sparse=False))
        assert eigenvalue_ds[1].item() == pytest.approx(1.0)

    def test_matches_second_order_perturbation_theory(self):
        eigenvalue, eigenvector = _unperturbed_pair()
        eigenvalue_ds, _ = eigenpair_derivatives(eigenvalue, eigenvector, _stiffness(sparse=False))
        assert eigenvalue_ds[2].item() == pytest.approx(-2.0)

    def test_eigenvector_first_derivative(self):
        eigenvalue, eigenvector = _unperturbed_pair()
        _, eigenvector_ds = eigenpair_derivatives(eigenvalue, eigenvector, _stiffness(sparse=False))
        assert np.allclose(np.ravel(eigenvector_ds[1]), [0.0, 1.0])

    def test_eigenvector_second_derivative(self):
        eigenvalue, eigenvector = _unperturbed_pair()
        _, eigenvector_ds = eigenpair_derivatives(eigenvalue, eigenvector, _stiffness(sparse=False))
        assert np.allclose(np.ravel(eigenvector_ds[2]), [-1.0, 4.0])

    def test_taylor_converges_at_the_truncation_rate(self):
        eigenvalue, eigenvector = _unperturbed_pair()
        eigenvalue_ds, _ = eigenpair_derivatives(eigenvalue, eigenvector, _stiffness(sparse=False))

        steps = [0.2, 0.1, 0.05]
        errors = [
            abs(eigenvalue_ds.evaluate_taylor(step).item()
                - np.linalg.eigvalsh(EVALUATION + step * FIRST_DERIVATIVE).min())
            for step in steps
        ]
        assert min(errors) > 1e-12
        rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        assert all(2.7 < rate < 3.3 for rate in rates)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_dense_and_sparse_agree(self, sparse):
        eigenvalue, eigenvector = _unperturbed_pair()
        reference, _ = eigenpair_derivatives(eigenvalue, eigenvector, _stiffness(sparse=False))
        result, _ = eigenpair_derivatives(eigenvalue, eigenvector, _stiffness(sparse=sparse))
        assert all(np.allclose(a, b) for a, b in zip(reference, result, strict=True))

    @pytest.mark.parametrize("sparse", [False, True])
    def test_output_is_always_dense(self, sparse):
        eigenvalue, eigenvector = _unperturbed_pair()
        eigenvalue_ds, eigenvector_ds = eigenpair_derivatives(eigenvalue, eigenvector, _stiffness(sparse=sparse))
        assert all(isinstance(element, np.ndarray) for element in eigenvalue_ds)
        assert all(isinstance(element, np.ndarray) for element in eigenvector_ds)


class TestConvergenceWithEveryOrderPopulated:
    evaluation = np.diag([1.0, 2.0, 3.5])
    coefficients = (
        np.array([[0.4, -0.3, 0.2], [-0.3, 0.1, 0.5], [0.2, 0.5, -0.2]]),
        np.array([[0.2, 0.1, -0.4], [0.1, -0.3, 0.2], [-0.4, 0.2, 0.6]]),
        np.array([[-0.5, 0.2, 0.1], [0.2, 0.4, -0.3], [0.1, -0.3, 0.2]]),
        np.array([[0.3, -0.1, 0.2], [-0.1, 0.5, 0.1], [0.2, 0.1, -0.4]]),
    )

    def _matrix_at(self, step):
        first, second, third, fourth = self.coefficients
        return (self.evaluation + step * first + step ** 2 / 2 * second
                + step ** 3 / 6 * third + step ** 4 / 24 * fourth)

    def _eigenvalue_ds(self):
        stiffness_ds = DerivativeSeries((self.evaluation,) + self.coefficients)
        unperturbed = np.linalg.eigh(self.evaluation)
        eigenvalue_ds, _ = eigenpair_derivatives(
            unperturbed.eigenvalues[0], unperturbed.eigenvectors[:, 0:1], stiffness_ds
        )
        return eigenvalue_ds

    @pytest.mark.parametrize("max_order", [2, 3, 4])
    def test_truncation_converges_at_its_own_rate(self, max_order):
        truncated = self._eigenvalue_ds().truncate(max_order)
        steps = [0.08, 0.04, 0.02]
        errors = [
            abs(truncated.evaluate_taylor(step).item() - np.linalg.eigvalsh(self._matrix_at(step)).min())
            for step in steps
        ]
        assert min(errors) > 1e-12
        rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        assert all(abs(rate - (max_order + 1)) < 0.3 for rate in rates)


class TestMassMatrixPaths:
    identity = DerivativeSeries((np.eye(2),))
    identity_with_zero_derivatives = DerivativeSeries((np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))))

    @pytest.mark.parametrize(
        "mass_mat_ds",
        [
            pytest.param(None, id="default"),
            pytest.param(identity, id="length one"),
            pytest.param(identity_with_zero_derivatives, id="length three"),
        ],
    )
    def test_the_identity_gives_the_same_result_however_it_is_passed(self, mass_mat_ds):
        eigenvalue, eigenvector = _unperturbed_pair()
        stiffness_ds = _stiffness(sparse=False)
        reference = eigenpair_derivatives(
            eigenvalue, eigenvector, stiffness_ds, self.identity_with_zero_derivatives
        )
        result = eigenpair_derivatives(eigenvalue, eigenvector, stiffness_ds, mass_mat_ds)
        for reference_ds, result_ds in zip(reference, result, strict=True):
            assert all(np.allclose(a, b) for a, b in zip(reference_ds, result_ds, strict=True))
