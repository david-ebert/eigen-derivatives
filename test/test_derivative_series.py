import math

import numpy as np
import scipy.sparse as sp
import pytest

from eigen_derivatives import DerivativeSeries


def _is_equal(x, ref) -> bool:
    diff = ref - x
    if hasattr(diff, "nnz"):
        out = diff.nnz == 0
    else:
        out = np.all(diff == 0)
    return out


class TestDerivativeSeries:
    TUPLE_NP = pytest.param((np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))), id="tuple numpy")
    LIST_NP = pytest.param([np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))], id="list numpy")
    TUPLE_SP = pytest.param((sp.eye(2, format='csc'), sp.csc_matrix((2, 2)), sp.csc_matrix((2, 2))), id="tuple scipy")
    INPUT_VARIANTS = [TUPLE_NP, LIST_NP, TUPLE_SP]

    @pytest.mark.parametrize("l", INPUT_VARIANTS, ids=lambda val: val.__name__)
    def test_basic_construct_iter_get(self, l):
        ds = DerivativeSeries(l)
        for ref, element in zip(l, ds):
            assert type(element) is type(ref)
            assert _is_equal(element, ref)
        assert _is_equal(ds[0], l[0])
        with pytest.raises(IndexError, match="tuple index out of range"):
            print(ds[len(ds)])

    @pytest.mark.parametrize("l", INPUT_VARIANTS)
    def test_repr(self, l):
        ds = DerivativeSeries(l)
        assert "DerivativeSeries(" + repr(tuple(l)) + ")" == repr(ds)

    def test_raises_different_size(self):
        with pytest.raises(ValueError, match="same shape"):
            DerivativeSeries((np.eye(2), np.zeros((2, 1)), np.zeros((2, 2))))

    def test_raises_bad_input_type(self):
        with pytest.raises(TypeError):
            DerivativeSeries(("This", "does", "not", "compute."))
        with pytest.raises(TypeError):
            DerivativeSeries((0, 1, 2, 3))

    def test_raises_array(self):
        ds = DerivativeSeries((np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))))
        with pytest.raises(TypeError, match="not an array"):
            print(np.asarray(ds))

    def test_horner(self):
        ds = DerivativeSeries((np.array(3), np.array(5), np.array(7)))
        assert math.isclose(ds.evaluate_taylor(0), 3.0)
        assert math.isclose(ds.evaluate_taylor(1), 11.5)
        assert math.isclose(ds.evaluate_taylor(11), 481.5)
        assert math.isclose(ds.evaluate_taylor(-1), 1.5)

    def test_truncate_keeps_the_lower_orders(self):
        ds = DerivativeSeries((np.array(3), np.array(5), np.array(7)))
        shortened = ds.truncate(1)
        assert len(shortened) == 2
        assert isinstance(shortened, DerivativeSeries)
        assert _is_equal(shortened[0], ds[0])
        assert _is_equal(shortened[1], ds[1])

    def test_truncate_drops_a_nan_tail(self):
        ds = DerivativeSeries((np.array(3.0), np.array(5.0), np.array(np.nan)))
        assert np.isnan(ds.evaluate_taylor(1))
        assert math.isclose(ds.truncate(1).evaluate_taylor(1), 8.0)

    def test_truncate_to_the_evaluation_only(self):
        ds = DerivativeSeries((np.array(3), np.array(5)))
        assert len(ds.truncate(0)) == 1

    def test_truncate_rejects_orders_outside_the_series(self):
        ds = DerivativeSeries((np.array(3), np.array(5)))
        with pytest.raises(ValueError, match="outside the available range"):
            ds.truncate(2)
        with pytest.raises(ValueError, match="outside the available range"):
            ds.truncate(-1)

    def test_raises_on_mixed_element_types(self):
        with pytest.raises(TypeError, match="same type"):
            DerivativeSeries((sp.eye(2, format="csc"), np.zeros((2, 2))))
        with pytest.raises(TypeError, match="same type"):
            DerivativeSeries((sp.eye(2, format="csc"), sp.csr_matrix((2, 2))))

    @pytest.mark.parametrize("l", INPUT_VARIANTS, ids=lambda val: val.__name__)
    def test_pad_with_zeros_keeps_type_and_shape(self, l):
        padded = DerivativeSeries(l).pad_with_zeros(5)
        assert len(padded) == 6
        assert all(type(element) is type(l[0]) for element in padded)
        assert all(element.shape == l[0].shape for element in padded)

    def test_pad_with_zeros_appends_zeros(self):
        padded = DerivativeSeries((np.eye(2), np.ones((2, 2)))).pad_with_zeros(3)
        assert np.all(padded[2] == 0)
        assert np.all(padded[3] == 0)
        assert padded[2] is not padded[3]

    def test_pad_with_zeros_keeps_sparse_sparse(self):
        padded = DerivativeSeries((sp.eye(2, format="csc"),)).pad_with_zeros(2)
        assert all(sp.issparse(element) for element in padded)
        assert padded[2].nnz == 0

    def test_pad_with_zeros_does_not_change_the_series_value(self):
        ds = DerivativeSeries((np.array(3.0), np.array(5.0)))
        assert math.isclose(ds.pad_with_zeros(4).evaluate_taylor(2.0), ds.evaluate_taylor(2.0))

    def test_pad_with_zeros_rejects_shortening(self):
        ds = DerivativeSeries((np.array(3), np.array(5), np.array(7)))
        with pytest.raises(ValueError, match="use truncate"):
            ds.pad_with_zeros(1)

    def test_evaluate_taylor_never_returns_an_integer_dtype(self):
        ds = DerivativeSeries((np.eye(2, dtype=int), np.eye(2, dtype=int), np.eye(2, dtype=int)))
        assert ds.evaluate_taylor(0.5).dtype == np.float64
        assert ds.evaluate_taylor(2).dtype == np.float64

    def test_evaluate_taylor_keeps_single_precision(self):
        ds = DerivativeSeries((np.eye(2, dtype=np.float32),) * 3)
        assert ds.evaluate_taylor(0.5).dtype == np.float32

    def test_evaluate_taylor_promotes_a_complex_lower_order(self):
        ds = DerivativeSeries((np.eye(2) * 1j, np.eye(2), np.eye(2)))
        assert ds.evaluate_taylor(0.5).dtype == np.complex128

    def test_evaluate_taylor_promotes_a_complex_step(self):
        ds = DerivativeSeries((np.eye(2), np.eye(2), np.eye(2)))
        assert ds.evaluate_taylor(0.5j).dtype == np.complex128
