"""Tests for the near-duplicate precision estimate, on made-up lables."""

import pandas as pd
import pytest

from measure_near_duplicate_precision import label_counts, precision_table


def labels(rows: dict[int, tuple[int, int]]) -> pd.DataFrame:
    """Label counts per distance: {distance: (labeled, positives)}."""
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["labeled", "positives"]
    ).rename_axis("distance")


def table_for(pairs: dict[int, int], known: dict[int, int], labeled) -> pd.DataFrame:
    """precision_table on plain dicts, indexed by distance; same_class = pairs."""
    return precision_table(
        pd.Series(pairs), pd.Series(known), labels(labeled), pd.Series(pairs)
    ).set_index("distance")


REVIEW = pd.DataFrame(
    {
        "drawn_by": ["dhash", "dhash phash", "phash"],
        "dhash_distance": [3, 3, 3],
        "phash_distance": [None, 4, 4],
        "label": ["copy", "copy", "unrelated"],
    }
)


def test_bins_weighted_by_pair_count():
    # A small clean bin must not count as much as a large noisy one
    table = table_for({0: 10, 1: 1000}, {0: 0}, {0: (10, 10), 1: (10, 1)})
    assert table.loc[0, "precision_pct"] == 100.0
    assert table.loc[1, "precision_pct"] == 10.0
    # (10 + 1000 * 0.1) / 1010, not the average of the two bins (55.0)
    assert table.loc[1, "precision_cumulative_pct"] == 10.9


def test_known_copies_count_fully_sample_scales_the_rest():
    # Bin 0: 62 known copies, 8 of 10 labeled pairs positive among the other 36
    table = table_for({0: 98}, {0: 62}, {0: (10, 8)})
    assert table.loc[0, "estimated_copies"] == 90.8  # 62 + 36 * 0.8
    assert table.loc[0, "precision_pct"] == 92.7  # 90.8 / 98


def test_each_hash_counts_only_its_own_draw():
    dhash = label_counts(REVIEW, "dhash")
    assert dhash.loc[3, "labeled"] == 2  # the phash-only row is left out
    assert dhash.loc[3, "positives"] == 2
    phash = label_counts(REVIEW, "phash")
    assert phash.loc[4, "labeled"] == 2  # "dhash phash" counts for both
    assert phash.loc[4, "positives"] == 1


def test_unsure_is_neither_positive_nor_negative():
    unsure = pd.DataFrame(
        {"drawn_by": ["dhash"], "dhash_distance": [3], "label": ["unsure"]}
    )
    dhash = label_counts(pd.concat([REVIEW, unsure]), "dhash")
    assert dhash.loc[3, "labeled"] == 2
    assert dhash.loc[3, "positives"] == 2


def test_bin_without_labels_raises():
    # Without the check, cumsum would silently skip the unlabeled bin
    with pytest.raises(ValueError, match="No labels"):
        table_for({0: 10, 1: 50}, {0: 0}, {0: (10, 10)})
