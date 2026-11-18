"""
Compute aHash, dHash and pHash for every image, from one decode per image.
Pixels after convert("RGB"), EXIF orientation not applied (same rule as
check_integrity.py).

Hashes are stored as 16-char hex strings: a 64-bit hash can exceed int64.
When loading, read them as strings (dtype=str) and convert with int(h, 16)
or np.uint64.

Columns: image_id, ahash, dhash, phash (empty if the file failed to decode)
Output: outputs/perceptual_hashes.csv
With --sample N: outputs/perceptual_hashes_sample.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, ImageFile

from image_dataset_audit.dataset import OUTPUTS, list_images, sample_images
from image_dataset_audit.parallel import parallel_map
from image_dataset_audit.perceptual_hash import dhash

# Truncated files must raise an error, never be silently padded.
# Set at import so each worker process gets it too.
ImageFile.LOAD_TRUNCATED_IMAGES = False


def compute_hashes(path: Path) -> dict:
    """aHash, dHash, and pHash of one image, as 16-char hex. Never raises."""
    row = {
        "image_id": f"{path.parent.name}/{path.stem}",
        "ahash": None,
        "dhash": None,
        "phash": None,
    }
    try:
        with Image.open(path) as im:
            img = im.convert("RGB")
        row["ahash"] = str(imagehash.average_hash(img))
        row["dhash"] = f"{dhash(img):016x}"
        row["phash"] = str(imagehash.phash(img))
    except Exception:
        row["ahash"] = None
        row["dhash"] = None
        row["phash"] = None
    return row


def parse_args() -> argparse.Namespace:
    """Command-line options; exits with a usage error on invalid values."""
    parser = argparse.ArgumentParser(
        description="aHash, dHash, and pHash for every Food-101 image."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="hash a random sample of N files (fixed seed) instead of all; "
        "writes outputs/perceptual_hashes_sample.csv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes (default: 1; 0 = all CPUs)",
    )
    parser.add_argument(
        "--max-fail-rate",
        type=float,
        default=0.01,
        help="exit with an error if more than this fraction of files fail "
        "(default: 0.01)",
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

    files = list_images() if args.sample is None else sample_images(args.sample)

    start = time.perf_counter()
    rows = parallel_map(
        compute_hashes, files, workers=workers, desc="Hashing", unit="file"
    )
    elapsed = time.perf_counter() - start

    cols = ["image_id", "ahash", "dhash", "phash"]
    df = pd.DataFrame(rows)[cols].sort_values("image_id", ignore_index=True)

    name = (
        "perceptual_hashes_sample.csv"
        if args.sample is not None
        else "perceptual_hashes.csv"
    )
    per_image_path = OUTPUTS / name
    OUTPUTS.mkdir(exist_ok=True)
    df.to_csv(per_image_path, index=False)

    failed = int(df[["ahash", "dhash", "phash"]].isna().any(axis=1).sum())
    fail_rate = failed / len(df)
    if fail_rate > args.max_fail_rate:
        sys.exit(
            f"\n{fail_rate:.1%} of files failed "
            f"(limit {args.max_fail_rate:.1%}). Check for a bug before using "
            f"{per_image_path}."
        )

    ms_per_file = elapsed / len(df) * 1000
    print(f"\n{len(df)} files in {elapsed:.1f} s ({ms_per_file:.2f} ms per file)")
    print(f"Failed to decode: {failed}")
    print(f"Saved: {per_image_path}")


if __name__ == "__main__":
    main()
