import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]  # repo root
DATA = ROOT / "data" / "food-101"
REPORTS = ROOT / "reports"


def load_split(split: str) -> pd.DataFrame:
    """
    Return one row per image listed in meta.
    """
    with open(DATA / "meta" / f"{split}.json") as f:
        mapping = json.load(f)  # {class_name: [image_id, ...]}

    rows = []

    for class_name, image_ids in mapping.items():
        for image_id in image_ids:
            rows.append({"image_id": image_id, "class": class_name, "split": split})
    return pd.DataFrame(rows)


def scan_actual_files(images_dir: Path) -> tuple[set[str], Counter[str], int]:
    """
    Scan through all files on images folder.
    - Image IDs
    - extension counts
    - number of files
    """
    ids: set[str] = set()
    exts: Counter[str] = Counter()
    n_files = 0
    for p in tqdm(images_dir.glob("*/*"), desc="Scanning files", unit="file"):
        if not p.is_file():
            continue
        n_files += 1
        exts[p.suffix] += 1
        ids.add(f"{p.parent.name}/{p.stem}")
    return ids, exts, n_files


def main() -> None:
    df = pd.concat([load_split("train"), load_split("test")], ignore_index=True)

    # check 1. per-class counts
    class_count = pd.crosstab(df["class"], df["split"])
    dup_in_class = int(df.duplicated(subset=["image_id", "split"]).sum())

    # check 2. class mismatch
    class_mismatch = int((df["image_id"].str.split("/").str[0] != df["class"]).sum())

    # check 3. IDs that appear in both train and test
    train_ids = set(df.loc[df["split"] == "train", "image_id"])
    test_ids = set(df.loc[df["split"] == "test", "image_id"])

    overlap = train_ids & test_ids

    # check 4. compare meta with the actual files
    meta_ids = train_ids | test_ids
    actual_ids, exts, n_files = scan_actual_files(DATA / "images")
    missing = meta_ids - actual_ids
    unlisted = actual_ids - meta_ids

    # One value per check, saved so README numbers come from reports/
    summary = {
        "classes": len(class_count),
        "images_listed": len(df),
        "train_listed": len(train_ids),
        "test_listed": len(test_ids),
        "train_per_class_min": int(class_count["train"].min()),
        "train_per_class_max": int(class_count["train"].max()),
        "test_per_class_min": int(class_count["test"].min()),
        "test_per_class_max": int(class_count["test"].max()),
        "listed_twice_in_split": dup_in_class,
        "in_both_train_and_test": len(overlap),
        "actual_files": n_files,
        "actual_unique_ids": len(actual_ids),
        "listed_but_missing": len(missing),
        "exists_but_not_listed": len(unlisted),
        "class_key_id_prefix_mismatch": class_mismatch,
    }
    # One row per extension, e.g. "ext_.jpg"
    summary.update({f"ext_{ext}": n for ext, n in sorted(exts.items())})

    REPORTS.mkdir(exist_ok=True)
    class_count.to_csv(REPORTS / "counts.csv")
    summary_df = pd.DataFrame(summary.items(), columns=["check", "value"])
    summary_df.to_csv(REPORTS / "counts_summary.csv", index=False)

    print(summary_df.to_string(index=False))
    if overlap:
        print(f"First overlapping IDs: {sorted(overlap)[:10]}")
    print(f"\nSaved: {REPORTS / 'counts.csv'}, {REPORTS / 'counts_summary.csv'}")


if __name__ == "__main__":
    main()
