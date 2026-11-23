"""Tests for the near-duplicate candidates stage, on small synthetic hashes."""

import numpy as np
import pandas as pd
import pytest

from find_near_duplicate_candidates import (
    class_codes,
    compare_all_pairs,
    same_class_pairs,
)

N = 20  # images in the random-hash tests


@pytest.fixture
def random_hashes() -> np.ndarray:
    """N random 64-bit hashes, fixed seed."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2**64, size=N, dtype=np.uint64)


@pytest.fixture
def random_classes() -> np.ndarray:
    """N class codes from 3 classes, fixed seed."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 3, size=N)


def test_radius_keeps_close_pairs_with_right_indices():
    """Only pairs within the radius are kept, with the right row numbers."""
    h = np.array([0b0000_0000, 0b1111_0000, 0b0000_0111, 0b1111_0011], dtype=np.uint64)
    classes = np.zeros(len(h), dtype=np.int64)  # not used by this test
    # distances: (0,1) 4, (0,2) 3, (0,3) 6, (1,2) 7, (1,3) 2, (2,3) 5
    _, _, (a, b, d) = compare_all_pairs(h, classes, radius=3)
    pairs = list(zip(a.tolist(), b.tolist(), d.tolist(), strict=True))
    # (0,2) sits exactly at the radius; (0,1) is one above it.
    # (1,3) has i = 1, j = 1: a wrong offset would point to another row.
    assert pairs == [(0, 2, 3), (1, 3, 2)]


def test_each_pair_once(random_hashes, random_classes):
    """Every pair is compared exactly once: a < b, no repeats, n(n-1)/2 in total."""
    _, _, (a, b, _) = compare_all_pairs(random_hashes, random_classes, radius=64)
    pairs = list(zip(a.tolist(), b.tolist(), strict=True))
    assert len(pairs) == N * (N - 1) // 2
    assert all(x < y for x, y in pairs)  # no self-pairs, one direction only
    assert len(set(pairs)) == len(pairs)  # no repeats


def test_no_radius_gives_counts_only(random_hashes, random_classes):
    """radius=None: counts for every pair, no candidate pairs."""
    counts, _, pairs = compare_all_pairs(random_hashes, random_classes, radius=None)
    assert pairs is None
    assert counts.sum() == N * (N - 1) // 2


def test_same_class_counted_at_their_distance():
    """Only same-class pairs land in same_counts, in their distance's row."""
    h = np.array([0b00, 0b01, 0b11, 0b00], dtype=np.uint64)
    classes = class_codes(pd.Series(["pie/1", "pie/2", "taco/3", "taco/4"]))
    # pairs: (0,1) d1 same, (0,2) d2, (0,3) d0, (1,2) d1, (1,3) d1, (2,3) d2 same
    counts, same_counts, _ = compare_all_pairs(h, classes, radius=None)
    assert counts[:3].tolist() == [1, 3, 2]
    assert same_counts[:3].tolist() == [0, 1, 1]
    assert same_counts.sum() == same_class_pairs(classes) == 2
