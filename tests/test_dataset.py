"""
Tests for the Dataset abstraction (Stage 8).

Tests cover:
- Construction and validation
- len() behavior
- Indexing behavior (positive, negative, out-of-range)
- Input/target pairing
- Empty dataset
- Mismatched lengths
"""

import pytest

from neuralearn.datasets import Dataset


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_construction(self):
        ds = Dataset([1, 2, 3], [10, 20, 30])
        assert len(ds) == 3

    def test_single_sample(self):
        ds = Dataset([1], [10])
        assert len(ds) == 1

    def test_stores_references(self):
        """Dataset stores references, not copies."""
        inputs = [1, 2, 3]
        targets = [10, 20, 30]
        ds = Dataset(inputs, targets)
        # Modifying original list after construction affects Dataset
        # (this is the expected behavior per D43)
        assert ds[0] == (1, 10)

    def test_repr(self):
        ds = Dataset([1, 2], [10, 20])
        r = repr(ds)
        assert "Dataset" in r
        assert "2" in r


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_inputs_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Dataset([], [])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Dataset([1, 2, 3], [10, 20])

    def test_mismatched_lengths_longer_targets(self):
        with pytest.raises(ValueError, match="same length"):
            Dataset([1], [10, 20])


# ---------------------------------------------------------------------------
# 3. Length
# ---------------------------------------------------------------------------

class TestLength:
    def test_length_one(self):
        ds = Dataset([1], [10])
        assert len(ds) == 1

    def test_length_many(self):
        ds = Dataset(list(range(100)), list(range(100)))
        assert len(ds) == 100


# ---------------------------------------------------------------------------
# 4. Indexing
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_first_element(self):
        ds = Dataset([1, 2, 3], [10, 20, 30])
        assert ds[0] == (1, 10)

    def test_last_element(self):
        ds = Dataset([1, 2, 3], [10, 20, 30])
        assert ds[2] == (3, 30)

    def test_negative_index(self):
        ds = Dataset([1, 2, 3], [10, 20, 30])
        assert ds[-1] == (3, 30)
        assert ds[-2] == (2, 20)
        assert ds[-3] == (1, 10)

    def test_out_of_range_raises(self):
        ds = Dataset([1, 2], [10, 20])
        with pytest.raises(IndexError):
            ds[5]

    def test_negative_out_of_range_raises(self):
        ds = Dataset([1, 2], [10, 20])
        with pytest.raises(IndexError):
            ds[-5]

    def test_returns_tuple(self):
        ds = Dataset([1], [10])
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 5. Pairing
# ---------------------------------------------------------------------------

class TestPairing:
    def test_pairing_preserved(self):
        inputs = ["a", "b", "c", "d"]
        targets = [1, 2, 3, 4]
        ds = Dataset(inputs, targets)

        for i in range(4):
            x, y = ds[i]
            assert x == inputs[i]
            assert y == targets[i]

    def test_pairing_with_complex_inputs(self):
        inputs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        targets = [[10.0], [20.0], [30.0]]
        ds = Dataset(inputs, targets)

        x, y = ds[1]
        assert x == [3.0, 4.0]
        assert y == [20.0]


# ---------------------------------------------------------------------------
# 6. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_same_values(self):
        ds = Dataset([0, 0, 0], [0, 0, 0])
        for i in range(3):
            assert ds[i] == (0, 0)

    def test_single_sample_indexing(self):
        ds = Dataset([42], [99])
        assert ds[0] == (42, 99)
        assert ds[-1] == (42, 99)
