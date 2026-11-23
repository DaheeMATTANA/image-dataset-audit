"""Time all-pairs Hamming distances on 5,000 dHashes: Python loop vs NumPy.

Run directly: each case runs in a new process, so no case inherits another's
memory state:

    uv run python experiments/time_hamming_case.py

Both methods count the pairs per distance (0 to 64), and the counts must match.
The runs are saved to reports/experiment_hamming_loop.csv.
"""

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from image_dataset_audit.dataset import OUTPUTS, REPORTS

N = 5000
SEED = 42
HASH = "dhash"
METHODS = ["python", "numpy"]
REPEATS = 3
PAIRS_101K = 101_000 * 100_999 // 2
PAIRS_10M = 10_000_000 * 9_999_999 // 2


def load_hashes(n: int, seed: int) -> list[int]:
    """n hashes from random rows of outputs/perceptual_hashes.csv, as Python ints."""
    df = pd.read_csv(OUTPUTS / "perceptual_hashes.csv", dtype=str).dropna()
    sample = df[HASH].sample(n, random_state=seed)
    return [int(h, 16) for h in sample]


def counts_python(hashes: list[int]) -> list[int]:
    """Pairs per distance 0-64: pure Python double loop, one pair at a time."""
    counts = [0] * 65
    n = len(hashes)
    for i in range(n):
        a = hashes[i]
        for j in range(i + 1, n):
            counts[(a ^ hashes[j]).bit_count()] += 1
    return counts


def counts_numpy(h: np.ndarray) -> list[int]:
    """Pairs per distance 0-64: one row at a time, XOR against the rows after it."""
    counts = np.zeros(65, dtype=np.int64)
    for i in range(len(h) - 1):
        d = np.bitwise_count(h[i] ^ h[i + 1 :])
        counts += np.bincount(d, minlength=65)
    return counts.tolist()


def measure(method: str) -> dict:
    """Time one method on the same N hashes; loading is not timed."""
    hashes = load_hashes(N, SEED)
    if method == "python":
        start = time.perf_counter()
        counts = counts_python(hashes)
    else:
        h = np.array(hashes, dtype=np.uint64)
        start = time.perf_counter()
        counts = counts_numpy(h)
    seconds = time.perf_counter() - start
    pairs = N * (N - 1) // 2
    return {
        "method": method,
        "n_images": N,
        "pairs": pairs,
        "seconds": round(seconds, 3),
        "pairs_per_s": round(pairs / seconds),
        "counts": counts,
    }


def run_case(method: str) -> dict:
    """Run one method in a new process and return its result."""
    out = subprocess.run(
        [sys.executable, __file__, method],
        stdout=subprocess.PIPE,  # stderr not captured: a crash shows its traceback
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


if __name__ == "__main__":
    if len(sys.argv) > 1:  # new process started by run_case: measure one method
        print(json.dumps(measure(sys.argv[1])))
    else:
        # Methods alternate (python, numpy, python, ...) so slow drift hits both.
        rows = [run_case(m) | {"repeat": r} for r in range(REPEATS) for m in METHODS]
        if any(row["counts"] != rows[0]["counts"] for row in rows):
            sys.exit("Counts differ between runs: the two methods disagree.")
        table = pd.DataFrame(rows).drop(columns="counts")
        print(table.to_string(index=False))

        best = table.groupby("method")["pairs_per_s"].median()
        for method, rate in best.items():
            print(
                f"{method}: {rate:,.0f} pairs/s (median) -> "
                f"101k: {PAIRS_101K / rate / 60:,.1f} min, "
                f"10M: {PAIRS_10M / rate / 3600:,.0f} h"
            )
        out = REPORTS / "experiment_hamming_loop.csv"
        table.to_csv(out, index=False)
        print(f"Saved: {out}")
