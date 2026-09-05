import numpy as np
import pytest
import scipy.linalg as sla
from scipy.optimize import linear_sum_assignment

from eigen_derivatives import (
    DerivativeSeries,
    eigenpair_derivatives,
    polarization_derivatives,
    polarize,
    polarized_eigenvectors,
)

DIRECTION = np.pi / 4
STEP_X = np.cos(DIRECTION)
STEP_Y = np.sin(DIRECTION)


def _zeros(n):
    return np.zeros((n, n))


def _example(n, n_ev, x0, y0, dxK, dyK, dx2K=None, dy2K=None, dx3K=None, dy3K=None, dx4K=None, dy4K=None):
    coefficients = [c if c is not None else _zeros(n) for c in (dx2K, dy2K, dx3K, dy3K, dx4K, dy4K)]
    dx2K, dy2K, dx3K, dy3K, dx4K, dy4K = coefficients
    base = 2.0 * np.eye(n)

    def matrix_at(x, y):
        return (base
                + x * dxK + x ** 2 / 2 * dx2K + x ** 3 / 6 * dx3K + x ** 4 / 24 * dx4K
                + y * dyK + y ** 2 / 2 * dy2K + y ** 3 / 6 * dy3K + y ** 4 / 24 * dy4K)

    stiffness_ds = DerivativeSeries((
        matrix_at(x0, y0),
        STEP_X * dxK + STEP_Y * dyK,
        STEP_X ** 2 * dx2K + STEP_Y ** 2 * dy2K,
        STEP_X ** 3 * dx3K + STEP_Y ** 3 * dy3K,
        STEP_X ** 4 * dx4K + STEP_Y ** 4 * dy4K,
    ))
    along_path = lambda t: matrix_at(x0 + STEP_X * t, y0 + STEP_Y * t)
    return stiffness_ds, n_ev, along_path


CROSS_CROSS = _example(
    3, 3, 0.0, 0.0,
    dxK=np.ones((3, 3)), dyK=_zeros(3),
    dx3K=2 * np.diag([0.0, 1.0, -1.0]), dy3K=2 * np.array([[0.0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    dx4K=6 * np.diag([0.0, 1.0, -1.0]), dy4K=6 * np.array([[0.0, 0, 0], [0, 0, 1], [0, 1, 0]]),
)
CROSS_DEFLECT = _example(
    3, 3, 0.0, 0.0,
    dxK=np.ones((3, 3)), dyK=_zeros(3),
    dx2K=2 * np.diag([0.0, 1.0, -1.0]), dy2K=2 * np.array([[0.0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    dx3K=6 * np.diag([0.0, 1.0, -1.0]), dy3K=6 * np.array([[0.0, 0, 0], [0, 0, 1], [0, 1, 0]]),
)
CROSS_FRISWELL = _example(
    2, 2, 0.0, 0.0,
    dxK=np.array([[1.0, 0.0], [0.0, -1.0]]), dyK=np.array([[0.0, 1.0], [1.0, 0.0]]),
)
DEFLECT = _example(
    2, 2, 0.0, 0.0,
    dxK=_zeros(2), dyK=_zeros(2),
    dx2K=np.array([[1.0, 0.0], [0.0, -1.0]]), dy2K=np.array([[0.0, 1.0], [1.0, 0.0]]),
)
PAIR_CROSS = _example(
    4, 4, 0.0, 0.0,
    dxK=np.array([[1.0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]), dyK=_zeros(4),
    dx2K=2 * np.array([[1.0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
    dy2K=2 * np.array([[0.0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
)
NONDEGENERATE = _example(
    2, 1, 0.25, -0.25,
    dxK=np.array([[1.0, 0.0], [0.0, -1.0]]), dyK=np.array([[0.0, 1.0], [1.0, 0.0]]),
)

EXPECTED_ORDER = {
    "cross cross": (CROSS_CROSS, np.array([[3, 3, 1], [3, 3, 1], [1, 1, 1]])),
    "cross deflect": (CROSS_DEFLECT, np.array([[2, 2, 1], [2, 2, 1], [1, 1, 1]])),
    "cross friswell": (CROSS_FRISWELL, np.ones((2, 2), dtype=int)),
    "deflect": (DEFLECT, 2 * np.ones((2, 2), dtype=int)),
    "pair cross": (PAIR_CROSS, np.array([[2, 2, 1, 1], [2, 2, 1, 1], [1, 1, 2, 2], [1, 1, 2, 2]])),
}


def _highest_finite_order(series):
    return max(order for order in range(len(series)) if np.isfinite(series[order]).all())


def _match_branches(predicted_eigenvectors, exact_eigenvectors):
    overlap = np.abs(predicted_eigenvectors.T @ exact_eigenvectors)
    _, matched = linear_sum_assignment(-overlap)
    return matched, overlap.max(axis=1).min()


def _workflow(example):
    stiffness_ds, n_ev, along_path = example
    unperturbed = np.linalg.eigh(stiffness_ds[0])
    eigenvalue_ds, eigenvector_ds = eigenpair_derivatives(
        unperturbed.eigenvalues[0], unperturbed.eigenvectors[:, :n_ev], stiffness_ds
    )
    initial_polarization, polarization_order = polarize(eigenvalue_ds)
    polarization_ds, polarized_eigenvalue_ds = polarization_derivatives(
        eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order
    )
    polarized_eigenvector_ds = polarized_eigenvectors(eigenvector_ds, polarization_ds)
    return polarization_order, polarized_eigenvalue_ds, along_path, n_ev, polarized_eigenvector_ds


class TestPolarizationOrder:
    @pytest.mark.parametrize("name", list(EXPECTED_ORDER))
    def test_matches_the_documented_splitting(self, name):
        example, expected = EXPECTED_ORDER[name]
        polarization_order, _, _, _, _ = _workflow(example)
        assert np.array_equal(polarization_order, expected)

    @pytest.mark.parametrize("name", list(EXPECTED_ORDER))
    def test_order_is_symmetric_with_ones_off_the_degenerate_blocks(self, name):
        example, _ = EXPECTED_ORDER[name]
        polarization_order, _, _, _, _ = _workflow(example)
        assert np.array_equal(polarization_order, polarization_order.T)
        assert polarization_order.min() >= 1


class TestPolarizedEigenvaluesAlongThePath:
    @pytest.mark.parametrize("name", list(EXPECTED_ORDER))
    def test_taylor_follows_the_exact_eigenvalues(self, name):
        example, _ = EXPECTED_ORDER[name]
        _, polarized_eigenvalue_ds, along_path, n_ev, _ = _workflow(example)
        for step in (0.05, 0.1):
            approximation = np.sort(np.ravel(polarized_eigenvalue_ds.evaluate_taylor(step)))
            exact = np.sort(np.linalg.eigvalsh(along_path(step)))[:n_ev]
            assert np.allclose(approximation, exact, atol=1e-5)


class TestNonDegenerateExample:
    def test_eigenvalue_derivatives_and_taylor(self):
        stiffness_ds, n_ev, along_path = NONDEGENERATE
        unperturbed = np.linalg.eigh(stiffness_ds[0])
        eigenvalue_ds, _ = eigenpair_derivatives(
            unperturbed.eigenvalues[0], unperturbed.eigenvectors[:, :n_ev], stiffness_ds
        )
        for step in (0.02, 0.05):
            exact = np.linalg.eigvalsh(along_path(step)).min()
            assert eigenvalue_ds.evaluate_taylor(step).item() == pytest.approx(exact, abs=1e-5)


class TestBranchIdentity:
    @pytest.mark.parametrize("name", list(EXPECTED_ORDER))
    @pytest.mark.parametrize("step", [0.1, -0.1])
    def test_each_branch_matches_its_own_eigenpair(self, name, step):
        example, _ = EXPECTED_ORDER[name]
        _, polarized_eigenvalue_ds, along_path, _, polarized_eigenvector_ds = _workflow(example)

        predicted_values = np.ravel(
            polarized_eigenvalue_ds.truncate(_highest_finite_order(polarized_eigenvalue_ds)).evaluate_taylor(step)
        )
        predicted_eigenvectors = (
            polarized_eigenvector_ds.truncate(_highest_finite_order(polarized_eigenvector_ds)).evaluate_taylor(step)
        )
        exact = np.linalg.eigh(along_path(step))

        matched, weakest_overlap = _match_branches(predicted_eigenvectors, exact.eigenvectors)
        assert weakest_overlap > 0.9
        assert np.allclose(predicted_values, exact.eigenvalues[matched], atol=1e-4)


class TestGeneralizedProblem:
    stiffness = (2.0 * np.eye(2), np.array([[1.0, -1.0], [-1.0, -1.0]]), np.zeros((2, 2)))
    mass = (
        np.eye(2),
        np.array([[0.3, 0.1], [0.1, -0.2]]),
        np.array([[0.1, 0.05], [0.05, 0.1]]),
    )
    steps = [0.2, 0.1, 0.05, 0.025]

    def _at(self, coefficients, step):
        evaluation, first, second = coefficients
        return evaluation + step * first + step ** 2 / 2 * second

    def _workflow(self):
        stiffness_ds = DerivativeSeries(self.stiffness)
        mass_ds = DerivativeSeries(self.mass)
        unperturbed_eigenvalues, unperturbed_eigenvectors = sla.eigh(self.stiffness[0], self.mass[0])
        eigenvalue_ds, eigenvector_ds = eigenpair_derivatives(
            unperturbed_eigenvalues[0], unperturbed_eigenvectors, stiffness_ds, mass_ds
        )
        initial_polarization, polarization_order = polarize(eigenvalue_ds)
        _, polarized_eigenvalue_ds = polarization_derivatives(
            eigenvalue_ds, eigenvector_ds, initial_polarization, polarization_order, mass_ds
        )
        return polarized_eigenvalue_ds, polarization_order

    def test_the_mass_matrix_stays_positive_definite(self):
        for step in self.steps + [-step for step in self.steps]:
            assert np.linalg.eigvalsh(self._at(self.mass, step)).min() > 0

    def test_the_unperturbed_eigenvectors_are_mass_orthonormal(self):
        _, eigenvectors = sla.eigh(self.stiffness[0], self.mass[0])
        assert np.allclose(eigenvectors.T @ self.mass[0] @ eigenvectors, np.eye(2))

    def test_polarization_order_is_one(self):
        _, polarization_order = self._workflow()
        assert np.array_equal(polarization_order, np.ones((2, 2)))

    def test_taylor_converges_against_the_generalized_eigenvalues(self):
        polarized_eigenvalue_ds, _ = self._workflow()
        errors = []
        for step in self.steps:
            approximation = np.sort(np.ravel(polarized_eigenvalue_ds.evaluate_taylor(step)))
            exact = np.sort(sla.eigh(self._at(self.stiffness, step), self._at(self.mass, step), eigvals_only=True))
            errors.append(np.abs(approximation - exact).max())

        assert min(errors) > 1e-12
        rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        assert all(abs(rate - 3.0) < 0.3 for rate in rates)
