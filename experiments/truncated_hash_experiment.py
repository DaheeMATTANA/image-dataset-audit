"""How many bits does a hash need before chance collisions appear?

Keeps the first N bits of file_sha256 and pixel_sha256 (outputs/integrity.csv;
N from 20 to 256, one hex character holds 4 bits), counts image pairs that share
them, and splits them into identical images (all 256 bits match) and chance
collisions. Adds the expected number of chance collisions at this size, 10M and
1B images.

Fewer bits on the same images behave like 32 bits on a larger dataset (chance
collisions depend on pairs / 2**N), so the 20-28 bit rows check the formula on
large measured counts before it is applied at 10M and 1B.

Writes reports/experiment_truncated_hash.csv.

    uv run python experiments/truncated_hash_experiment.py
"""

import pandas as pd

from image_dataset_audit.dataset import OUTPUTS, REPORTS

HASHES = ["file_sha256", "pixel_sha256"]
BITS = [20, 24, 28, 32, 64, 256]  # measured on this dataset
SCALE_BITS = [32, 64, 256]  # widths for the expected counts at SCALES
SCALES = [10_000_000, 1_000_000_000]  # dataset sizes for the expected counts


def colliding_pairs(values: pd.Series) -> int:
    """Number of row pairs sharing a value."""
    sizes = values.value_counts()
    return int((sizes * (sizes - 1) // 2).sum())


def expected_pairs(n: int, bits: int) -> float:
    """Expected number of chance collisions among n hashes of `bits` bits."""
    return n * (n - 1) / 2 / 2**bits


def truncation_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per hash, width and dataset size.

    Columns: hash, bits, images, pairs_sharing_bits, identical_pairs,
    chance_pairs, expected_chance_pairs. Measured columns are empty (None)
    for the SCALES rows.
    """
    rows = []
    for key in HASHES:
        hashes = df[key]
        n = len(hashes)
        identical = colliding_pairs(hashes)  # all 256 bits match
        for bits in BITS:
            shared = colliding_pairs(hashes.str[: bits // 4])  # 4 bits per hex char
            chance = shared - identical
            if bits == 256:
                assert chance == 0, f"{key}: full hashes must give no chance pairs"
            rows.append(
                {
                    "hash": key,
                    "bits": bits,
                    "images": n,
                    "pairs_sharing_bits": shared,
                    "identical_pairs": identical,
                    "chance_pairs": chance,
                    "expected_chance_pairs": expected_pairs(n, bits),
                }
            )
        for size in SCALES:
            for bits in SCALE_BITS:
                rows.append(
                    {
                        "hash": key,
                        "bits": bits,
                        "images": size,
                        "pairs_sharing_bits": None,
                        "identical_pairs": None,
                        "chance_pairs": None,
                        "expected_chance_pairs": expected_pairs(size, bits),
                    }
                )
    measured = ["pairs_sharing_bits", "identical_pairs", "chance_pairs"]
    return pd.DataFrame(rows).astype({column: "Int64" for column in measured})


def readable(x: float) -> str:
    """Expected counts of 1 or more as numbers; below 1, the chance of seeing even
    one chance pair, as "1 in N" (N rounded to 3 significant digits)."""
    if x >= 1:
        return f"{x:,.2f}"
    if x >= 1e-12:
        return f"1 in {float(f'{1 / x:.3g}'):,.0f}"
    return "< 1 in a trillion"


def main() -> None:
    df = pd.read_csv(OUTPUTS / "integrity.csv", usecols=["image_id", *HASHES])
    df = df.dropna(subset=HASHES)  # no pixel hash if a file failed to decode
    table = truncation_table(df)
    print(table.to_string(index=False, formatters={"expected_chance_pairs": readable}))
    out = REPORTS / "experiment_truncated_hash.csv"
    table.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
