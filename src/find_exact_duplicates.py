"""Find exact duplicates: images that share a file hash or a pixel hash.

Reads outputs/integrity.csv (written by check_integrity.py), groups images by
file_sha256 and then by pixel_sha256, and classifies every group of two or more
images by split: within train, within test, or across train and test (leakage).
Prints the counts per kind and writes one row per group to
outputs/exact_duplicates_file.csv and outputs/exact_duplicates_pixel.csv.

    uv run python src/find_exact_duplicates.py
"""

import pandas as pd

from dataset_utils import OUTPUTS

HASHES = {"file": "file_sha256", "pixel": "pixel_sha256"}


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
            kind=("split", split_kind),
            n_classes=("class", "nunique"),
            image_ids=("image_id", " ".join),
            classes=("class", " ".join),
            splits=("split", " ".join),
        )
        .reset_index()
        .sort_values(["kind", "size"], ascending=[True, False])
    )


def summarize(groups: pd.DataFrame, n_images: int) -> pd.DataFrame:
    """Groups and images per kind; extra copies = images a dedup would remove."""
    summary = groups.groupby("kind").agg(
        groups=("size", "size"),
        images=("size", "sum"),
    )
    summary["extra_copies"] = summary["images"] - summary["groups"]
    summary["extra_copies_pct"] = (100 * summary["extra_copies"] / n_images).round(3)
    return summary


def main() -> None:
    df = pd.read_csv(
        OUTPUTS / "integrity.csv",
        usecols=["image_id", "split", "class", "file_sha256", "pixel_sha256"],
    )
    for name, key in HASHES.items():
        hashed = df.dropna(subset=[key])  # no pixel hash if a file failed to decode
        groups = duplicate_groups(hashed, key)
        print(f"\n== {key}: {len(groups)} groups of identical {name}s")
        print(summarize(groups, len(hashed)).to_string())
        out = OUTPUTS / f"exact_duplicates_{name}.csv"
        groups.to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
