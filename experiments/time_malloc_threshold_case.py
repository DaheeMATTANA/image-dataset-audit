"""Time the slow k=5000 case with glibc malloc thresholds fixed one at a time.

Each case runs in a new process (run_case): glibc reads MALLOC_* variables only
at process start. One run per case: fault counts are identical from run to run,
and the speed gaps are far larger than run-to-run noise. The runs are saved to
reports/experiment_malloc_threshold.csv.

    uv run python experiments/time_malloc_threshold_case.py
"""

import os
import sys

from time_serial_case import report, run_case

MIB = 2**20
MMAP_16 = {"MALLOC_MMAP_THRESHOLD_": str(16 * MIB)}
TRIM_128 = {"MALLOC_TRIM_THRESHOLD_": str(128 * MIB)}
# Smaller trim values around what one image frees (about 2.2 MB), to find the
# smallest that still avoids page faults. Paired with mmap 16 MiB.
SMALL_TRIMS_MIB = [1, 2, 4, 8]

CASES = [
    {},  # default
    TRIM_128,  # trim only
    MMAP_16,  # mmap only
    TRIM_128 | MMAP_16,  # both, mmap 16 MiB
    TRIM_128 | {"MALLOC_MMAP_THRESHOLD_": str(64 * MIB)},  # both, mmap 64 MiB
    *({"MALLOC_TRIM_THRESHOLD_": str(t * MIB)} | MMAP_16 for t in SMALL_TRIMS_MIB),
]


def threshold_mb(env: dict[str, str], name: str) -> float | None:
    """A MALLOC_* threshold in decimal MB (10^6 bytes), or None if not set."""
    return round(int(env[name]) / 1e6, 1) if name in env else None


if __name__ == "__main__":
    # run_case adds to the parent's environment: an exported MALLOC_* would leak in.
    if any(k.startswith("MALLOC_") for k in os.environ):
        sys.exit("Unset MALLOC_* variables first (see README benchmark commands).")
    rows = [
        run_case(env=env)
        | {
            "mmap_threshold_mb": threshold_mb(env, "MALLOC_MMAP_THRESHOLD_"),
            "trim_threshold_mb": threshold_mb(env, "MALLOC_TRIM_THRESHOLD_"),
        }
        for env in CASES
    ]
    report(rows, by=["mmap_threshold_mb", "trim_threshold_mb"], name="malloc_threshold")
