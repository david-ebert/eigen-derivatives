import math

import numpy as np
import pytest
import scipy.sparse as sp

from eigen_derivatives.utils import (
    _bilinear_form,
    _multiindex_total_order,
    _multinomial_coefficient,
    _with_coefficients,
)

ORDERS = list(range(0, 9))
LENGTHS = list(range(1, 7))


class TestMultiIndexEnumeration:
    @pytest.mark.parametrize("total_order", ORDERS, ids=lambda value: f"order {value}")
    @pytest.mark.parametrize("length", LENGTHS, ids=lambda value: f"length {value}")
    def test_the_shape_matches_the_number_of_compositions(self, total_order, length):
        rows = _multiindex_total_order(total_order, length)
        assert rows.shape == (math.comb(total_order + length - 1, length - 1), length)

    @pytest.mark.parametrize("total_order", ORDERS, ids=lambda value: f"order {value}")
    @pytest.mark.parametrize("length", LENGTHS, ids=lambda value: f"length {value}")
    def test_every_row_is_non_negative_and_sums_to_the_order(self, total_order, length):
        rows = _multiindex_total_order(total_order, length)
        assert np.all(rows >= 0)
        assert np.all(rows.sum(axis=1) == total_order)

    @pytest.mark.parametrize("total_order", ORDERS, ids=lambda value: f"order {value}")
    @pytest.mark.parametrize("length", LENGTHS, ids=lambda value: f"length {value}")
    def test_no_multi_index_appears_twice(self, total_order, length):
        rows = _multiindex_total_order(total_order, length)
        assert len({tuple(row) for row in rows.tolist()}) == len(rows)

    def test_the_degenerate_arguments_keep_their_shapes(self):
        assert _multiindex_total_order(3, 0).shape == (0, 0)
        assert _multiindex_total_order(-1, 3).shape == (0, 3)
        assert np.array_equal(_multiindex_total_order(3, 1), [[3]])


class TestMultinomialCoefficients:
    @pytest.mark.parametrize("total_order", ORDERS, ids=lambda value: f"order {value}")
    @pytest.mark.parametrize("length", LENGTHS, ids=lambda value: f"length {value}")
    def test_the_coefficients_obey_the_multinomial_theorem(self, total_order, length):
        # summing over all multi-indices of one order reproduces length ** order
        rows = _multiindex_total_order(total_order, length)
        assert sum(_multinomial_coefficient(rows)) == length ** total_order

    @pytest.mark.parametrize(
        "multi_index, expected",
        [((3, 0), 1), ((2, 1), 3), ((1, 1), 2), ((2, 2), 6), ((1, 1, 1), 6), ((2, 1, 1), 12)],
        ids=["3+0", "2+1", "1+1", "2+2", "1+1+1", "2+1+1"],
    )
    def test_known_values(self, multi_index, expected):
        assert _multinomial_coefficient(np.array([multi_index])) == [expected]

    def test_a_single_multi_index_may_be_one_dimensional(self):
        assert _multinomial_coefficient(np.array([2, 1])) == [3]

    def test_an_empty_list_gives_no_coefficients(self):
        assert _multinomial_coefficient(np.empty((0, 3), dtype=int)) == []

    def test_the_coefficients_stay_exact_beyond_the_float_range(self):
        # 23! is the first factorial that float64 cannot represent exactly
        assert _multinomial_coefficient(np.array([[23, 1]])) == [24]
        assert _multinomial_coefficient(np.array([[12, 12]])) == [math.comb(24, 12)]


class TestBilinearForm:
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])

    def test_it_matches_the_explicit_product_for_real_input(self):
        left = np.array([1.0, 2.0])
        right = np.array([3.0, -1.0])
        assert _bilinear_form(left, right, middle=self.matrix) == pytest.approx(
            left.T @ self.matrix @ right
        )

    def test_an_omitted_middle_stands_for_the_identity(self):
        left = np.array([1.0, 2.0])
        right = np.array([3.0, -1.0])
        assert _bilinear_form(left, right) == pytest.approx(left @ right)
        assert _bilinear_form(left, right) == pytest.approx(
            _bilinear_form(left, right, middle=np.eye(2))
        )

    def test_it_conjugates_the_left_argument(self):
        vector = np.array([1.0 + 2.0j, 3.0 - 1.0j])
        norm_squared = _bilinear_form(vector, vector)
        assert norm_squared.imag == pytest.approx(0.0)
        assert norm_squared.real == pytest.approx(np.linalg.norm(vector) ** 2)

    def test_a_hermitian_middle_gives_a_real_quadratic_form(self):
        hermitian = np.array([[2.0, 1.0 + 1.0j], [1.0 - 1.0j, 3.0]])
        vector = np.array([1.0 + 2.0j, 3.0 - 1.0j])
        value = _bilinear_form(vector, vector, middle=hermitian)
        assert value.imag == pytest.approx(0.0)

    def test_it_works_on_blocks_and_returns_a_matrix(self):
        left = np.arange(6.0).reshape(3, 2)
        right = np.arange(6.0, 12.0).reshape(3, 2)
        middle = np.eye(3) * 2.0
        result = _bilinear_form(left, right, middle=middle)
        assert result.shape == (2, 2)
        assert np.allclose(result, left.T @ middle @ right)

    def test_a_sparse_middle_is_accepted(self):
        left = np.arange(6.0).reshape(3, 2)
        right = np.arange(6.0, 12.0).reshape(3, 2)
        middle = sp.diags_array([1.0, 2.0, 3.0], format="csc")
        result = _bilinear_form(left, right, middle=middle)
        assert np.allclose(result, left.T @ middle.toarray() @ right)

    def test_the_middle_is_keyword_only(self):
        left = np.array([1.0, 2.0])
        with pytest.raises(TypeError):
            _bilinear_form(left, left, self.matrix)


class TestWithCoefficients:
    def test_every_row_is_paired_with_its_own_coefficient(self):
        rows = _multiindex_total_order(4, 3)
        expected = _multinomial_coefficient(rows)
        pairs = list(_with_coefficients(rows))
        assert len(pairs) == len(rows)
        for (coefficient, multi_index), row, reference in zip(pairs, rows, expected, strict=True):
            assert coefficient == reference
            assert np.array_equal(multi_index, row)
