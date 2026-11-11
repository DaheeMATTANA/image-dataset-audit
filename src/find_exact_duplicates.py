"""Find exact duplicates: images that share a file hash or a pixel hash.

Reads outputs/integrity.csv (written by check_integrity.py), groups images by
file_sha256 and then by pixel_sha256, and classifies every group of two or more
images by split: within train, within test, or across train and test (leakage).
Prints the counts per kind, writes one row per group to
outputs/exact_duplicates_file.csv and outputs/exact_duplicates_pixel.csv, and
writes the counts per hash and kind to reports/exact_duplicates_summary.csv.

    uv run python src/find_exact_duplicates.py
"""

import pandas as pd

from dataset_utils import OUTPUTS, REPORTS

HASHES = {"file": "file_sha256", "pixel": "pixel_sha256"}
KINDS = ["within train", "within test", "across train and test"]


def split_kind(splits: pd.Series) -> str:
    """Where a group's images come from: one split, or both (leakage)."""
    found = set(splits)
    if found == {"train", "test"}:
        return "across train and test"
    (split,) = found
    return f"within {split}"


def duplicate_groups(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """One row per group of two or more images sharing the value in `key`."""
    shared = df[df.duplicated(key, keep=False)].sort_values("image_id")
    return (
        shared.groupby(key)
        .agg(
            size=("image_id", "size"),
            n_test=("split", lambda s: int((s == "test").sum())),
            kind=("split", split_kind),
            n_classes=("class", "nunique"),
            image_ids=("image_id", " ".join),
            classes=("class", " ".join),
            splits=("split", " ".join),
        )
        .reset_index()
        .sort_values(["kind", "size"], ascending=[True, False])
    )


def summarize(groups: pd.DataFrame, n_images: int, n_test: int) -> pd.DataFrame:
    """Per kind, plus an "all" row: groups, images, extra copies (images a dedup
    would remove) and test images in these groups, with their percentages."""
    summary = (
        groups.groupby("kind")
        .agg(
            groups=("size", "size"),
            images=("size", "sum"),
            test_images=("n_test", "sum"),
        )
        .reindex(KINDS, fill_value=0)
    )
    summary.loc["all"] = summary.sum()
    summary["extra_copies"] = summary["images"] - summary["groups"]
    summary["extra_copies_pct"] = (100 * summary["extra_copies"] / n_images).round(3)
    summary["test_images_pct"] = (100 * summary["test_images"] / n_test).round(3)
    return summary.rename_axis("kind").reset_index()


def main() -> None:
    df = pd.read_csv(
        OUTPUTS / "integrity.csv",
        usecols=["image_id", "split", "class", "file_sha256", "pixel_sha256"],
    )
    summaries = []
    for name, key in HASHES.items():
        hashed = df.dropna(subset=[key])  # no pixel hash if a file failed to decode
        n_test = int((hashed["split"] == "test").sum())
        groups = duplicate_groups(hashed, key)
        summary = summarize(groups, len(hashed), n_test)
        print(f"\n== {key}: {len(groups)} groups of identical {name}s")
        print(summary.to_string(index=False))
        out = OUTPUTS / f"exact_duplicates_{name}.csv"
        groups.to_csv(out, index=False)
        print(f"Saved: {out}")
        summaries.append(summary.assign(hash=name))

    report = pd.concat(summaries)
    columns = [
        "hash",
        "kind",
        "groups",
        "images",
        "extra_copies",
        "extra_copies_pct",
        "test_images",
        "test_images_pct",
    ]
    out = REPORTS / "exact_duplicates_summary.csv"
    report[columns].to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
