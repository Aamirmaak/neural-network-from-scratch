"""
Tests for the DataLoader abstraction (Stage 8).

Tests cover:
- Construction and validation
- Batch size behavior
- Exact and partial division
- drop_last behavior
- Shuffle behavior (pairing preservation, reproducibility)
- Seed behavior
- len(loader)
- Reusable iteration
- Edge cases
"""

import pytest

from neuralearn.datasets import Dataset
from neuralearn.dataloaders import DataLoader


def _make_dataset(n=10):
    """Create a simple dataset with n samples."""
    inputs = [[float(i)] for i in range(n)]
    targets = [[float(i * 2)] for i in range(n)]
    return Dataset(inputs, targets)


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_construction(self):
        ds = _make_dataset(10)
        loader = DataLoader(ds, batch_size=3)
        assert loader.dataset is ds
        assert loader.batch_size == 3

    def test_none_dataset_raises(self):
        with pytest.raises(TypeError):
            DataLoader(None, batch_size=1)

    def test_zero_batch_size_raises(self):
        ds = _make_dataset(5)
        with pytest.raises(ValueError, match="batch_size"):
            DataLoader(ds, batch_size=0)

    def test_negative_batch_size_raises(self):
        ds = _make_dataset(5)
        with pytest.raises(ValueError, match="batch_size"):
            DataLoader(ds, batch_size=-1)

    def test_defaults(self):
        ds = _make_dataset(5)
        loader = DataLoader(ds)
        assert loader.batch_size == 1
        assert loader.shuffle is False
        assert loader.drop_last is False
        assert loader.seed is None

    def test_repr(self):
        ds = _make_dataset(5)
        loader = DataLoader(ds, batch_size=2, shuffle=True, seed=42)
        r = repr(loader)
        assert "DataLoader" in r
        assert "2" in r
        assert "True" in r


# ---------------------------------------------------------------------------
# 2. Length
# ---------------------------------------------------------------------------

class TestLength:
    def test_exact_division_no_drop(self):
        ds = _make_dataset(10)
        loader = DataLoader(ds, batch_size=5)
        assert len(loader) == 2

    def test_partial_batch_no_drop(self):
        ds = _make_dataset(10)
        loader = DataLoader(ds, batch_size=3)
        # ceil(10/3) = 4
        assert len(loader) == 4

    def test_exact_division_with_drop(self):
        ds = _make_dataset(10)
        loader = DataLoader(ds, batch_size=5, drop_last=True)
        assert len(loader) == 2

    def test_partial_batch_with_drop(self):
        ds = _make_dataset(10)
        loader = DataLoader(ds, batch_size=3, drop_last=True)
        # floor(10/3) = 3
        assert len(loader) == 3

    def test_batch_size_equals_dataset(self):
        ds = _make_dataset(5)
        loader = DataLoader(ds, batch_size=5)
        assert len(loader) == 1

    def test_batch_size_larger_than_dataset(self):
        ds = _make_dataset(3)
        loader = DataLoader(ds, batch_size=10)
        assert len(loader) == 1

    def test_batch_size_1(self):
        ds = _make_dataset(5)
        loader = DataLoader(ds, batch_size=1)
        assert len(loader) == 5


# ---------------------------------------------------------------------------
# 3. Batching
# ---------------------------------------------------------------------------

class TestBatching:
    def test_batch_size_1(self):
        ds = _make_dataset(4)
        loader = DataLoader(ds, batch_size=1)
        batches = list(loader)
        assert len(batches) == 4
        for batch_inputs, batch_targets in batches:
            assert len(batch_inputs) == 1
            assert len(batch_targets) == 1

    def test_batch_size_equals_dataset(self):
        ds = _make_dataset(4)
        loader = DataLoader(ds, batch_size=4)
        batches = list(loader)
        assert len(batches) == 1
        batch_inputs, batch_targets = batches[0]
        assert len(batch_inputs) == 4
        assert len(batch_targets) == 4

    def test_batch_size_larger_than_dataset(self):
        ds = _make_dataset(3)
        loader = DataLoader(ds, batch_size=10)
        batches = list(loader)
        assert len(batches) == 1
        assert len(batches[0][0]) == 3

    def test_exact_division(self):
        ds = _make_dataset(6)
        loader = DataLoader(ds, batch_size=3)
        batches = list(loader)
        assert len(batches) == 2
        for batch_inputs, batch_targets in batches:
            assert len(batch_inputs) == 3
            assert len(batch_targets) == 3

    def test_partial_final_batch(self):
        ds = _make_dataset(7)
        loader = DataLoader(ds, batch_size=3)
        batches = list(loader)
        assert len(batches) == 3
        assert len(batches[0][0]) == 3
        assert len(batches[1][0]) == 3
        assert len(batches[2][0]) == 1

    def test_batch_contents_correct(self):
        ds = _make_dataset(6)
        loader = DataLoader(ds, batch_size=3)
        batches = list(loader)

        # First batch: samples 0, 1, 2
        assert batches[0][0] == [[0.0], [1.0], [2.0]]
        assert batches[0][1] == [[0.0], [2.0], [4.0]]

        # Second batch: samples 3, 4, 5
        assert batches[1][0] == [[3.0], [4.0], [5.0]]
        assert batches[1][1] == [[6.0], [8.0], [10.0]]


# ---------------------------------------------------------------------------
# 4. drop_last
# ---------------------------------------------------------------------------

class TestDropLast:
    def test_drop_last_discards_partial(self):
        ds = _make_dataset(7)
        loader = DataLoader(ds, batch_size=3, drop_last=True)
        batches = list(loader)
        assert len(batches) == 2
        for batch_inputs, batch_targets in batches:
            assert len(batch_inputs) == 3

    def test_drop_last_exactDivision_noEffect(self):
        ds = _make_dataset(6)
        loader = DataLoader(ds, batch_size=3, drop_last=True)
        batches = list(loader)
        assert len(batches) == 2

    def test_drop_last_datasetSmallerThanBatch(self):
        ds = _make_dataset(2)
        loader = DataLoader(ds, batch_size=5, drop_last=True)
        batches = list(loader)
        assert len(batches) == 0

    def test_drop_last_singleSample(self):
        ds = _make_dataset(1)
        loader = DataLoader(ds, batch_size=1, drop_last=True)
        batches = list(loader)
        assert len(batches) == 1


# ---------------------------------------------------------------------------
# 5. No Shuffle
# ---------------------------------------------------------------------------

class TestNoShuffle:
    def test_order_preserved(self):
        ds = _make_dataset(6)
        loader = DataLoader(ds, batch_size=3)
        batches = list(loader)

        # Samples in order
        all_inputs = []
        for batch_inputs, _ in batches:
            all_inputs.extend(batch_inputs)
        assert all_inputs == [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]


# ---------------------------------------------------------------------------
# 6. Shuffle
# ---------------------------------------------------------------------------

class TestShuffle:
    def test_shuffle_preserves_pairing(self):
        """Shuffling must keep input[i] paired with target[i]."""
        inputs = [[float(i)] for i in range(10)]
        targets = [[float(i * 10)] for i in range(10)]
        ds = Dataset(inputs, targets)
        loader = DataLoader(ds, batch_size=5, shuffle=True, seed=42)

        batches = list(loader)
        all_inputs = []
        all_targets = []
        for bi, bt in batches:
            all_inputs.extend(bi)
            all_targets.extend(bt)

        # Each input must be paired with its correct target
        input_to_target = {}
        for x, y in zip(all_inputs, all_targets):
            input_to_target[x[0]] = y[0]

        for i in range(10):
            assert input_to_target[float(i)] == float(i * 10)

    def test_shuffle_changes_order(self):
        """Shuffling should (likely) change the order from sequential."""
        ds = _make_dataset(20)
        loader_no_shuffle = DataLoader(ds, batch_size=5, shuffle=False)
        loader_shuffle = DataLoader(ds, batch_size=5, shuffle=True, seed=42)

        batches_no = list(loader_no_shuffle)
        batches_yes = list(loader_shuffle)

        # Collect all inputs in order
        order_no = []
        for bi, _ in batches_no:
            order_no.extend(bi)
        order_yes = []
        for bi, _ in batches_yes:
            order_yes.extend(bi)

        # With seed=42 on 20 elements, order should differ
        assert order_no != order_yes

    def test_shuffled_all_samples_present(self):
        """All samples must appear exactly once after shuffling."""
        ds = _make_dataset(10)
        loader = DataLoader(ds, batch_size=3, shuffle=True, seed=42)

        all_inputs = []
        for batch_inputs, _ in loader:
            all_inputs.extend(batch_inputs)

        # All 10 samples present
        assert len(all_inputs) == 10
        sorted_vals = sorted([x[0] for x in all_inputs])
        assert sorted_vals == [float(i) for i in range(10)]


# ---------------------------------------------------------------------------
# 7. Seed Behavior
# ---------------------------------------------------------------------------

class TestSeed:
    def test_same_seed_same_order(self):
        """Two DataLoaders with the same seed produce the same initial ordering."""
        ds = _make_dataset(10)
        loader1 = DataLoader(ds, batch_size=3, shuffle=True, seed=42)
        loader2 = DataLoader(ds, batch_size=3, shuffle=True, seed=42)

        batches1 = list(loader1)
        batches2 = list(loader2)

        for (bi1, bt1), (bi2, bt2) in zip(batches1, batches2):
            assert bi1 == bi2
            assert bt1 == bt2

    def test_different_seed_different_order(self):
        """Different seeds should (likely) produce different orderings."""
        ds = _make_dataset(20)
        loader1 = DataLoader(ds, batch_size=5, shuffle=True, seed=42)
        loader2 = DataLoader(ds, batch_size=5, shuffle=True, seed=123)

        order1 = []
        for bi, _ in loader1:
            order1.extend(bi)
        order2 = []
        for bi, _ in loader2:
            order2.extend(bi)

        assert order1 != order2

    def test_no_seed_still_works(self):
        """No seed should still produce valid batches."""
        ds = _make_dataset(10)
        loader = DataLoader(ds, batch_size=3, shuffle=True)
        batches = list(loader)

        all_inputs = []
        for bi, _ in batches:
            all_inputs.extend(bi)
        assert len(all_inputs) == 10


# ---------------------------------------------------------------------------
# 8. Reusable Iteration
# ---------------------------------------------------------------------------

class TestReusableIteration:
    def test_double_iteration(self):
        ds = _make_dataset(6)
        loader = DataLoader(ds, batch_size=3)

        batches1 = list(loader)
        batches2 = list(loader)

        # Same results on both iterations
        assert len(batches1) == len(batches2)
        for (bi1, bt1), (bi2, bt2) in zip(batches1, batches2):
            assert bi1 == bi2
            assert bt1 == bt2

    def test_shuffle_reusable(self):
        """Shuffle iteration works multiple times with fresh permutations."""
        ds = _make_dataset(10)
        loader = DataLoader(ds, batch_size=3, shuffle=True, seed=42)

        batches1 = list(loader)
        batches2 = list(loader)

        # Both iterations should have all samples
        for batches in [batches1, batches2]:
            all_inputs = []
            for bi, _ in batches:
                all_inputs.extend(bi)
            assert len(all_inputs) == 10


# ---------------------------------------------------------------------------
# 9. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_sample(self):
        ds = _make_dataset(1)
        loader = DataLoader(ds, batch_size=1)
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0][0] == [[0.0]]
        assert batches[0][1] == [[0.0]]

    def test_batch_size_1_full_iteration(self):
        ds = _make_dataset(5)
        loader = DataLoader(ds, batch_size=1)
        batches = list(loader)
        assert len(batches) == 5
        for i, (bi, bt) in enumerate(batches):
            assert bi == [[float(i)]]
            assert bt == [[float(i * 2)]]
