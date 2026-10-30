"""
Profile every image in Food-101: one row per file, header-only reads.

Columns: image_id, split, class, width, height, long_side, format, mode,
size_bytes, error
Output: outputs/profile_images.csv (per image), reports/profile_summary.csv
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from dataset_utils import DATA, OUTPUTS, REPORTS, load_meta


def profile_file(path: Path) -> dict:
    """Header-only profile of one file.

    Never raises: errors go in the 'error' column.
    """
    row = {
        "image_id": f"{path.parent.name}/{path.stem}",
        "class": path.parent.name,
        "size_bytes": path.stat().st_size,
        "width": None,
        "height": None,
        "format": None,
        "mode": None,
        "error": None,
    }
    try:
        with Image.open(path) as im:
            row["width"], row["height"] = im.size
            row["format"] = im.format
            row["mode"] = im.mode
    except Exception as e:  # not an image, truncated header, etc.
        row["error"] = type(e).__name__
    return row


def build_profile(images_dir: Path) -> pd.DataFrame:
    """One row per file, with its split joined from meta/."""
    rows = [
        profile_file(p)
        for p in tqdm(images_dir.glob("*/*"), desc="Profiling", unit="file")
        if p.is_file()
    ]
    df = pd.DataFrame(rows)

    size_cols = ["width", "height", "long_side"]
    df["long_side"] = df[["width", "height"]].max(axis=1)
    df[size_cols] = df[size_cols].astype("Int64")

    meta = load_meta()[["image_id", "split"]]
    df = df.merge(meta, on="image_id", how="left")
    df["split"] = df["split"].fillna("unlisted")

    cols = [
        "image_id",
        "split",
        "class",
        "width",
        "height",
        "long_side",
        "format",
        "mode",
        "size_bytes",
        "error",
    ]

    return df[cols].sort_values("image_id", ignore_index=True)


def summarize(df: pd.DataFrame) -> dict:
    """One value per check."""
    readable = df[df["error"].isna()]
    summary = {
        "files": len(df),
        "unreadable": int(df["error"].notna().sum()),
        "long_side_is_512_pct": round(
            float((readable["long_side"] == 512).mean() * 100), 2
        ),
        "long_side_min": int(readable["long_side"].min()),
        "long_side_max": int(readable["long_side"].max()),
        "short_side_min": int(readable[["width", "height"]].min(axis=1).min()),
        "size_kb_mean": round(df["size_bytes"].mean() / 1000, 1),
        "size_kb_median": round(df["size_bytes"].median() / 1000, 1),
        "size_kb_min": round(df["size_bytes"].min() / 1000, 1),
        "size_kb_max": round(df["size_bytes"].max() / 1000, 1),
        "size_total_gb": round(df["size_bytes"].sum() / 1e9, 2),
    }

    # One row per value found
    for col in ["split", "format", "mode", "error"]:
        counts = df[col].value_counts()
        summary.update({f"{col}_{value}": int(n) for value, n in counts.items()})
    return summary


def main() -> None:
    df = build_profile(DATA / "images")
    summary = summarize(df)
    summary_df = pd.DataFrame(
        {
            "check": list(summary.keys()),
            "value": pd.Series(list(summary.values()), dtype="object"),
        }
    )

    profile_path = OUTPUTS / "profile_images.csv"
    summary_path = REPORTS / "profile_summary.csv"
    OUTPUTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(profile_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    problems = df[
        df["error"].notna() | (df["long_side"] != 512) | (df["mode"] != "RGB")
    ]
    if not problems.empty:
        print(f"\nFirst files worth a look ({len(problems)} total):")
        print(problems.head(10).to_string(index=False))
    print(f"\nSaved: {profile_path}, {summary_path}")


if __name__ == "__main__":
    main()
