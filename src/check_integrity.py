"""
Check that every image fully decodes, and hash its file bytes and its pixels.
Pixel hash = pixels after convert("RGB"), EXIF orientation not applied.

Columns: image_id, split, class, decode_ok, error_type, error_message,
file_sha256, pixel_sha256
Output: outputs/integrity.csv (per image), reports/integrity_summary.csv
With --sample N: outputs/integrity_sample.csv only (no report).
"""

import argparse
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageFile

from dataset_utils import DATA, OUTPUTS, REPORTS, load_meta
from parallel_utils import parallel_map

IMAGES = DATA / "images"
SEED = 42

# Truncated files must raise an error, never be silently padded.
# Set explicitly (not just checked) so the behavior never depends on defaults,
# and so each worker process that imports this module gets it too.
ImageFile.LOAD_TRUNCATED_IMAGES = False


def get_sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of the file bytes, read in chunks so memory use stays constant."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def get_sha256_pixels(arr: np.ndarray) -> str:
    """SHA-256 of the decoded image array (shape included).

    The caller passes the pixels after convert("RGB"), without applying EXIF
    orientation: what a typical training loader sees. So color mode is ignored
    (an L image equals an RGB image with three identical channels), alpha is
    dropped, and files differing only in their orientation tag hash the same.
    """
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr))
    return h.hexdigest()


def check_integrity(path: Path) -> dict:
    """Integrity signals for one file. Never raises: errors are recorded in the row."""
    row = {
        "image_id": f"{path.parent.name}/{path.stem}",
        "class": path.parent.name,
        "decode_ok": False,
        "error_type": None,
        "error_message": None,
        "file_sha256": None,
        "pixel_sha256": None,
    }
    try:
        row["file_sha256"] = get_sha256_file(path)
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        row["pixel_sha256"] = get_sha256_pixels(arr)
        row["decode_ok"] = True
    except Exception as e:
        row["error_type"] = type(e).__name__
        row["error_message"] = str(e)[:200]
    return row


def summarize(df: pd.DataFrame) -> dict:
    """One value per check."""
    summary = {
        "files": len(df),
        "decode_ok": int(df["decode_ok"].sum()),
        "decode_failed": int((~df["decode_ok"]).sum()),
        # pixel hashes depend on the JPEG decoder, so record its version
        "pillow_version": PIL.__version__,
    }
    for col in ["split", "error_type"]:
        counts = df[col].value_counts()
        summary.update({f"{col}_{value}": int(n) for value, n in counts.items()})
    return summary


def parse_args() -> argparse.Namespace:
    """Command-line options; exits with a usage error on invalid values."""
    parser = argparse.ArgumentParser(
        description="Decode check and SHA-256 hashes for every Food-101 image."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="check a random sample of N files (fixed seed) instead of all; "
        "writes outputs/integrity_sample.csv and no report",
    )
    parser.add_argument(
        "--max-fail-rate",
        type=float,
        default=0.01,
        help="stop without writing the report if more than this fraction of "
        "files fail to decode (default: 0.01)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes (default: 1; 0 = all CPUs)",
    )
    args = parser.parse_args()
    if args.sample is not None and args.sample <= 0:
        parser.error("--sample must be a positive integer")
    if args.workers < 0:
        parser.error("--workers must be 0 (all CPUs) or a positive integer")
    return args


def main() -> None:
    args = parse_args()
    workers = os.cpu_count() if args.workers == 0 else args.workers  # 0 = use all CPUs

    files = sorted(p for p in IMAGES.glob("*/*") if p.is_file())
    if not files:
        sys.exit(f"No files found under {IMAGES}")
    if args.sample is not None:
        files = random.Random(SEED).sample(files, min(args.sample, len(files)))

    start = time.perf_counter()
    rows = parallel_map(
        check_integrity, files, workers=workers, desc="Checking", unit="file"
    )
    elapsed = time.perf_counter() - start

    df = (
        pd.DataFrame(rows)
        .merge(load_meta()[["image_id", "split"]], on="image_id", how="left")
        .sort_values("image_id", ignore_index=True)
    )
    df["split"] = df["split"].fillna("unlisted")
    cols = [
        "image_id",
        "split",
        "class",
        "decode_ok",
        "error_type",
        "error_message",
        "file_sha256",
        "pixel_sha256",
    ]
    df = df[cols].sort_values("image_id", ignore_index=True)

    name = "integrity_sample.csv" if args.sample is not None else "integrity.csv"
    per_image_path = OUTPUTS / name
    OUTPUTS.mkdir(exist_ok=True)
    df.to_csv(per_image_path, index=False)

    summary = summarize(df)
    summary_df = pd.DataFrame(
        {
            "check": list(summary.keys()),
            "value": pd.Series(list(summary.values()), dtype="object"),
        }
    )
    print(summary_df.to_string(index=False))
    ms_per_file = elapsed / len(df) * 1000
    print(f"\n{len(df)} files in {elapsed:.1f} s ({ms_per_file:.2f} ms per file)")
    print(f"Saved: {per_image_path}")

    # Sanity check
    fail_rate = summary["decode_failed"] / len(df)
    if fail_rate > args.max_fail_rate:
        print("\nError types:")
        print(df["error_type"].value_counts().to_string())
        sys.exit(
            f"\n{fail_rate:.1%} of files failed to decode "
            f"(limit {args.max_fail_rate:.1%}). Check for a bug before trusting "
            f"this; report not written. Inspect {per_image_path}."
        )

    if args.sample is None:
        summary_path = REPORTS / "integrity_summary.csv"
        REPORTS.mkdir(exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
