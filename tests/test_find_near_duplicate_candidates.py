"""Tests for the near-duplicate candidates stage, on small synthetic hashes."""

import numpy as np
import pytest

from find_near_duplicate_candidates import compare_all_pairs

N = 20  # images in the random-hash tests

@pytest.fixture
def random_hashes() -> np.ndarray:
    """N random 64-bit hashes, fixed seed."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2**64, size=N, dtype=np.uint64)


def test_radius_keeps_close_pairs_with_right_indices():
    """Only pairs within the radius are kept, with the right row numbers."""
    h = np.array([0b0000_0000, 0b1111_0000, 0b0000_0111, 0b1111_0011], dtype=np.uint64)
    # distances: (0,1) 4, (0,2) 3, (0,3) 6, (1,2) 7, (1,3) 2, (2,3) 5
    _, (a, b, d) = compare_all_pairs(h, radius=3)
    pairs = list(zip(a.tolist(), b.tolist(), d.tolist(), strict=True))
    # (0,2) sits exactly at the radius; (0,1) is one above it.
    # (1,3) has i = 1, j = 1: a wrong offset would point to another row.
    assert pairs == [(0, 2, 3), (1, 3, 2)]


def test_each_pair_once(random_hashes):
    """Every pair is compared exactly once: a < b, no repeats, n(n-1)/2 in total."""
    counts, (a, b, _) = compare_all_pairs(random_hashes, radius=64)     # keep all
    pairs = list(zip(a.tolist(), b.tolist(), strict=True))
    assert len(pairs) == N * (N - 1) // 2
    assert all(x < y for x, y in pairs)     # no self-pairs, one direction only
    assert len(set(pairs)) == len(pairs)    # no repeats
    assert counts.sum() == N * (N - 1) // 2


def test_no_radius_gives_counts_only(random_hashes):
    """radius=None: counts for every pair, no candidate pairs."""
    counts, pairs = compare_all_pairs(random_hashes, radius=None)
    assert pairs is None
    assert counts.sum() == N * (N - 1) // 2