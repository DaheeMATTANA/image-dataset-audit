"""How many bytes does one entry of an in-memory hash index take?

Builds a Python dict from pixel hash to image for every image in
outputs/integrity.csv, for several key and value types, and measures the bytes
per entry: key object, value object and the dict's own hash-table slot. Projects
the total size at 10M and 1B images (bytes per entry x N).

Writes reports/experiment_index_entry_size.csv.

    uv run python experiments/index_entry_size_experiment.py
"""

import sys

import pandas as pd

from image_dataset_audit.dataset import OUTPUTS, REPORTS

SCALES = {"gb_at_10m": 10_000_000, "gb_at_1b": 1_000_000_000}


def entry_bytes(index: dict) -> dict:
    """Bytes per entry of `index`, split into key, value and hash-table slot.

    sys.getsizeof(index) counts only the dict's own table (the slots pointing to
    keys and values), so the key and value objects are added one by one.
    """
    n = len(index)
    key_bytes = sum(sys.getsizeof(key) for key in index) / n
    value_bytes = sum(sys.getsizeof(value) for value in index.values()) / n
    slot_bytes = sys.getsizeof(index) / n
    return {
        "key_bytes": round(key_bytes, 1),
        "value_bytes": round(value_bytes, 1),
        "slot_bytes": round(slot_bytes, 1),
        "bytes_per_entry": round(key_bytes + value_bytes + slot_bytes, 1),
    }


def entry_size_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per key and value type: bytes per entry and projected size."""
    hex_keys = list(df["pixel_sha256"])
    digests = [bytes.fromhex(h) for h in hex_keys]
    ids = list(df["image_id"])
    variants = {
        ("hex string", "ID string"): (hex_keys, ids),
        ("32-byte digest", "ID string"): (digests, ids),
        ("8-byte prefix", "ID string"): ([d[:8] for d in digests], ids),
        ("4-byte prefix", "ID string"): ([d[:4] for d in digests], ids),
        ("32-byte digest", "row number"): (digests, list(range(len(df)))),
    }
    rows = []
    for (key_type, value_type), (keys, values) in variants.items():
        index = dict(zip(keys, values, strict=True))  # one entry per unique key
        sizes = entry_bytes(index)
        projected = {
            name: round(sizes["bytes_per_entry"] * n / 1e9, 2)
            for name, n in SCALES.items()
        }
        rows.append(
            {
                "key": key_type,
                "value": value_type,
                "entries": len(index),
                **sizes,
                **projected,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(OUTPUTS / "integrity.csv", usecols=["image_id", "pixel_sha256"])
    df = df.dropna(subset=["pixel_sha256"])
    table = entry_size_table(df)
    print(table.to_string(index=False))
    out = REPORTS / "experiment_index_entry_size.csv"
    table.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
