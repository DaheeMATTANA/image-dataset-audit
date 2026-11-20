"""Count image pairs per Hamming distance and write near-duplicate candidates.

Every pair of images is compared once, for each perceptual hash. The counts per
distance (0-64) show how many pairs each radius would keep; the pairs within
RADIUS are written as candidates (aHash: counts only).

    uv run python src/find_near_duplicate_candidates.py                # all images
    uv run python src/find_near_duplicate_candidates.py --sample 5000  # quick check
"""

import argparse
import time

import numpy as np
import pandas as pd

from image_dataset_audit.dataset import OUTPUTS, REPORTS, SEED

HASHES = ["ahash", "dhash", "phash"]
RADIUS = {"dhash": 12, "phash": 12}  # aHash: counts only, no candidates


def load_hashes(sample: int | None) -> pd.DataFrame:
    """Hash table from the hash stage; rows with a failed decode are dropped."""
    df = pd.read_csv(OUTPUTS / "perceptual_hashes.csv", dtype=str).dropna()
    if sample:
        df = df.sample(sample, random_state=SEED)
    return df


def to_uint64(hex_hashes: pd.Series) -> np.ndarray:
    """16-char hex strings -> one uint64 per image."""
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def compare_all_pairs(h: np.ndarray, radius: int | None):
    """Pairs per distance 0-64, and the pairs within radius (None: no pairs)."""
    counts = np.zeros(65, dtype=np.int64)
    rows_a, rows_b, dists = [], [], []
    for i in range(len(h) - 1):
        d = np.bitwise_count(h[i] ^ h[i + 1 :])
        counts += np.bincount(d, minlength=65)
        if radius is not None:
            j = np.nonzero(d <= radius)[0]  # positions in d that are <= radius
            rows_a.append(np.full(len(j), i))  # image i, once per kept pair
            rows_b.append(i + 1 + j)  # the other image
            dists.append(d[j])  # their distances
    if radius is None:
        return counts, None
    return counts, (
        np.concatenate(rows_a),
        np.concatenate(rows_b),
        np.concatenate(dists),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, help="use N random images")
    args = parser.parse_args()

    df = load_hashes(args.sample)
    n = len(df)
    print(f"{n:,} images, {n * (n - 1) // 2:,} pairs per hash")

    counts = {}
    suffix = f"_sample{args.sample}" if args.sample else ""
    for name in HASHES:
        h = to_uint64(df[name])
        start = time.perf_counter()
        counts[name], pairs = compare_all_pairs(h, RADIUS.get(name))
        print(f"{name}: {time.perf_counter() - start:.1f} s")
        if counts[name].sum() != n * (n - 1) // 2:
            raise SystemExit(f"{name}: counts don't add up to the number of pairs")
        if pairs is not None:
            a, b, dist = pairs
            ids = df["image_id"].to_numpy()
            cands = pd.DataFrame(
                {"image_id_a": ids[a], "image_id_b": ids[b], "distance": dist}
            )
            out = OUTPUTS / f"near_duplicate_candidates_{name}{suffix}.csv"
            cands.to_csv(out, index=False)
            print(f"{name}: {len(cands):,} candidate pairs -> {out}")
    table = pd.DataFrame({"distance": range(65), **counts})
    out_dir = OUTPUTS if args.sample else REPORTS  # sample runs are checks, not reports
    out = out_dir / f"near_duplicate_distance_counts{suffix}.csv"
    table.to_csv(out, index=False)
    print(table.head(21).to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
