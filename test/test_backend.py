import numpy as np
import pytest
import scipy.sparse as sp

from eigen_derivatives import DerivativeSeries, eigenpair_derivatives
from eigen_derivatives.backend import Backend, DenseBackend, SparseBackend, get_backend
from eigen_derivatives.polarization import polarization_derivatives, polarize

SYMMETRIC_POSITIVE = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
SYMMETRIC_INDEFINITE = np.array([[1.0, 2.0, 0.0], [2.0, -3.0, 1.0], [0.0, 1.0, 0.0]])
UNSYMMETRIC = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
MATRICES = [SYMMETRIC_POSITIVE, SYMMETRIC_INDEFINITE, UNSYMMETRIC]
MATRIX_IDS = ["positive definite", "indefinite", "unsymmetric"]
RHS_SHAPES = [(3,), (3, 1), (3, 2)]


class TestBackendSelection:
    def test_a_dense_element_selects_the_dense_backend(self):
        assert isinstance(get_backend(np.eye(2)), DenseBackend)

    def test_a_sparse_element_selects_the_sparse_backend(self):
        assert isinstance(get_backend(sp.eye_array(2, format="csc")), SparseBackend)

    def test_the_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Backend()


class TestDenseBackend:
    @pytest.mark.parametrize("matrix", MATRICES, ids=MATRIX_IDS)
    @pytest.mark.parametrize("shape", RHS_SHAPES, ids=[str(s) for s in RHS_SHAPES])
    def test_the_factorized_solver_matches_a_plain_solve(self, matrix, shape):
        rhs = np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape)
        solver = DenseBackend().factorize(matrix)
        assert np.allclose(solver(rhs), np.linalg.solve(matrix, rhs))

    @pytest.mark.parametrize("matrix", MATRICES, ids=MATRIX_IDS)
    def test_the_solver_can_be_reused_for_several_right_hand_sides(self, matrix):
        solver = DenseBackend().factorize(matrix)
        for column in np.eye(3):
            assert np.allclose(matrix @ solver(column), column)

    def test_a_complex_system_is_solved(self):
        matrix = np.array([[2.0, 1.0 + 1.0j], [1.0 - 1.0j, 3.0]])
        rhs = np.array([1.0, 1.0j])
        solver = DenseBackend().factorize(matrix)
        assert np.allclose(matrix @ solver(rhs), rhs)

    def test_the_constructors_return_dense_arrays(self):
        backend = DenseBackend()
        assert isinstance(backend.zeros(2, 3), np.ndarray)
        assert isinstance(backend.eye(2), np.ndarray)
        assert backend.zeros(2, 3).shape == (2, 3)
        assert np.array_equal(backend.block([[np.eye(1), np.zeros((1, 1))]]), [[1.0, 0.0]])


class TestSparseBackend:
    @pytest.mark.parametrize("shape", RHS_SHAPES, ids=[str(s) for s in RHS_SHAPES])
    def test_both_solve_paths_keep_the_shape_of_the_right_hand_side(self, shape):
        matrix = sp.csc_array(SYMMETRIC_POSITIVE)
        rhs = np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape)
        backend = SparseBackend()
        assert np.shape(backend.solve(matrix, rhs)) == shape
        assert np.shape(backend.factorize(matrix)(rhs)) == shape

    @pytest.mark.parametrize("shape", RHS_SHAPES, ids=[str(s) for s in RHS_SHAPES])
    def test_the_factorized_solver_matches_the_dense_result(self, shape):
        rhs = np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape)
        solver = SparseBackend().factorize(sp.csc_array(SYMMETRIC_POSITIVE))
        assert np.allclose(solver(rhs), np.linalg.solve(SYMMETRIC_POSITIVE, rhs))

    def test_the_constructors_return_sparse_arrays(self):
        backend = SparseBackend()
        assert sp.issparse(backend.zeros(2, 3))
        assert sp.issparse(backend.eye(2))
        assert sp.issparse(backend.block([[backend.eye(1), backend.zeros(1, 1)]]))
        assert backend.zeros(2, 3).shape == (2, 3)


class TestDefaultFactorization:
    class _NoFactorization(Backend):
        """A backend that implements only the required operations."""

        def as_matrix(self, matrix):
            return np.asarray(matrix)

        def block(self, blocks):
            return np.block(blocks)

        def zeros(self, rows, cols):
            return np.zeros((rows, cols))

        def eye(self, dim):
            return np.eye(dim)

        def solve(self, matrix, rhs):
            return np.linalg.solve(matrix, rhs)

    def test_a_backend_without_a_factorization_still_solves(self):
        solver = self._NoFactorization().factorize(SYMMETRIC_POSITIVE)
        rhs = np.array([1.0, 2.0, 3.0])
        assert np.allclose(solver(rhs), np.linalg.solve(SYMMETRIC_POSITIVE, rhs))


class TestBackendInjection:
    """The public functions accept a backend so that a user can supply their own."""

    class _CountingBackend(DenseBackend):
        def __init__(self):
            self.factorizations = 0
            self.solves = 0

        def factorize(self, matrix):
            self.factorizations += 1
            return super().factorize(matrix)

        def solve(self, matrix, rhs):
            self.solves += 1
            return super().solve(matrix, rhs)

    STIFFNESS = np.diag([1.0, 2.0, 2.0, 5.0])
    PERTURBATION = np.diag([0.1, 0.3, -0.2, 0.4])
    MASS = np.eye(4)

    def _problem(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.STIFFNESS)
        return 2.0, eigenvectors[:, 1:3]

    def test_eigenpair_derivatives_uses_the_given_backend(self):
        eigenvalue, eigenvectors = self._problem()
        stiffness = DerivativeSeries((self.STIFFNESS, self.PERTURBATION))
        mass = DerivativeSeries((self.MASS,))
        backend = self._CountingBackend()

        with_backend = eigenpair_derivatives(eigenvalue, eigenvectors, stiffness, mass, backend=backend)
        without = eigenpair_derivatives(eigenvalue, eigenvectors, stiffness, mass)

        assert backend.factorizations == 1
        for given, default in zip(with_backend[0], without[0], strict=True):
            assert np.allclose(given, default)
        for given, default in zip(with_backend[1], without[1], strict=True):
            assert np.allclose(given, default)

    def test_polarization_derivatives_uses_the_given_backend(self):
        # a block that stays degenerate at first order and splits at second, so that the
        # branch which actually solves is reached
        eigenvalue_ds = DerivativeSeries(
            (np.zeros((3, 3)), np.diag([1.0, 1.0, 2.0]), np.diag([3.0, 7.0, 0.0]))
        )
        eigenvector_ds = DerivativeSeries((np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))))
        init_polarization, order = polarize(eigenvalue_ds)
        backend = self._CountingBackend()

        given = polarization_derivatives(
            eigenvalue_ds, eigenvector_ds, init_polarization, order, backend=backend
        )
        default = polarization_derivatives(
            eigenvalue_ds, eigenvector_ds, init_polarization, order
        )

        assert backend.solves > 0
        for one, other in zip(given[1], default[1], strict=True):
            assert np.allclose(one, other)

    def test_the_backend_argument_is_keyword_only(self):
        eigenvalue, eigenvectors = self._problem()
        stiffness = DerivativeSeries((self.STIFFNESS, self.PERTURBATION))
        with pytest.raises(TypeError):
            eigenpair_derivatives(eigenvalue, eigenvectors, stiffness, None, DenseBackend())
