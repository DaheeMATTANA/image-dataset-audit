"""Measure the precision of near-duplicate candidates from hand labels.

For each hash (dHash, pHash) and each Hamming distance up to the candidate
radius, estimates how many candidate pairs are real near-duplicates (labels
copy or near-copy). Each distance bin is estimated from the pairs that hash
drew for review (annotations/near_duplicate_review.csv). The exact pixel pairs
(outputs/exact_duplicates_pixel.csv) sit at distance 0 and count as known
copies. The cumulative columns give the precision of the rule "distance <= t".
Writes reports/near_duplicate_precision.csv.

    uv run python src/measure_near_duplicate_precision.py
"""

import pandas as pd

from find_near_duplicate_candidates import RADIUS
from image_dataset_audit.dataset import OUTPUTS, REPORTS, ROOT

REVIEW = ROOT / "annotations" / "near_duplicate_review.csv"
POSITIVE = ["copy", "near_copy"]
EXCLUDED = ["unsure"]  # neither positive nor negative


def exact_pair_count() -> int:
    """Pairs of images with identical pixels (a group of n gives n(n-1)/2)."""
    groups = pd.read_csv(OUTPUTS / "exact_duplicates_pixel.csv", usecols=["size"])
    sizes = groups["size"]
    return int((sizes * (sizes - 1) // 2).sum())


def label_counts(review: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per distance of hash `name`: labeled pairs and positives, using only
    the pairs this hash drew (a pair drawn by the other hash would bias it)."""
    own = review[review["drawn_by"].str.split().map(lambda hashes: name in hashes)]
    own = own[~own["label"].isin(EXCLUDED)]
    return (
        own.groupby(f"{name}_distance")
        .agg(
            labeled=("label", "size"),
            positives=("label", lambda s: int(s.isin(POSITIVE).sum())),
        )
        .rename_axis("distance")
    )


def precision_table(
    pairs: pd.Series, known: pd.Series, labels: pd.DataFrame, same_class: pd.Series
) -> pd.DataFrame:
    """Estimated copies and precision per distance, and for "distance <= t".

    pairs: all candidate pairs per distance, known: pairs known to be copies
    (not sampled); labels: labeled and positives per distance (label_counts).
    """
    table = pd.DataFrame(
        {"pairs": pairs, "same_class_pairs": same_class, "known_copies": known}
    ).fillna(0)
    table = table.rename_axis("distance")
    table = table.join(labels).fillna(0).astype(int)
    table = table[table["pairs"] > 0]  # pHash: odd distances are empty

    sampled = table["pairs"] - table["known_copies"]
    if ((sampled > 0) & (table["labeled"] == 0)).any():
        missing = table.index[(sampled > 0) & (table["labeled"] == 0)].tolist()
        raise ValueError(f"No labels for distances {missing}")

    share = (table["positives"] / table["labeled"]).where(sampled > 0, 0)
    copies = table["known_copies"] + sampled * share  # sample scaled to the bin
    table["estimated_copies"] = copies.round(1)
    # copies are almost always same-class (except label conflicts), so a ratio
    # well above 1 points to sampling noise from few labels
    same = table["same_class_pairs"]
    table["copies_per_same_class_pair"] = (copies / same).where(same > 0).round(2)
    table["precision_pct"] = (100 * copies / table["pairs"]).round(1)
    table["pairs_cumulative"] = table["pairs"].cumsum()
    table["precision_cumulative_pct"] = (
        100 * copies.cumsum() / table["pairs_cumulative"]
    ).round(1)
    return table.reset_index()


def main() -> None:
    counts = pd.read_csv(REPORTS / "near_duplicate_distance_counts.csv")
    counts = counts.set_index("distance")
    review = pd.read_csv(REVIEW)
    n_exact = exact_pair_count()

    tables = []
    for name, radius in RADIUS.items():
        pairs = counts.loc[:radius, name]
        same_class = counts.loc[:radius, f"{name}_same_class"]
        known = pd.Series({0: n_exact})  # identical pixels -> identical hash
        table = precision_table(pairs, known, label_counts(review, name), same_class)
        tables.append(table.assign(hash=name))
        print(f"\n{name} (radius {radius}):")
        print(table.to_string(index=False))

    result = pd.concat(tables)
    result = result[["hash", *result.columns.drop("hash")]]
    out = REPORTS / "near_duplicate_precision.csv"
    result.to_csv(out, index=False)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
