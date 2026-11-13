"""Time check_integrity in a serial loop, one case per new Python process.

Run directly to compare core types (CPU 4 = P-core, CPU 12 = E-core on this machine):

    uv run python experiments/time_serial_case.py

A new process per case means no case inherits another's memory state
(e.g. glibc malloc's). Each case runs REPEATS times; the runs are saved to
reports/experiment_core_type.csv.
"""

import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

from check_integrity import check_integrity
from image_dataset_audit.dataset import IMAGES, REPORTS
from image_dataset_audit.parallel import parallel_map

DEFAULTS = {"k": 5000, "n": 5000, "seed": 42, "cpu": None, "warmup": 0}
CORE_TYPE_CASES = [{"cpu": 4}, {"cpu": 12}, {}]
REPEATS = 3


def current_cpu() -> int:
    """CPU this process last ran on (field 39 of /proc/self/stat, Linux)."""
    stat = Path("/proc/self/stat").read_text()
    return int(stat.rsplit(")", 1)[1].split()[36])


def warmup_files(files: list[Path], sample: list[Path], size: int) -> list[Path]:
    """The timed files first, then untimed files in sorted order, up to size files."""
    timed = set(sample)
    return (sample + [p for p in files if p not in timed])[:size]


def measure(k: int, n: int, seed: int, cpu: int | None, warmup: int) -> dict:
    """Time check_integrity in a serial loop over n files.

    The files are the first n of random.Random(seed).sample(files, k). Also counts
    minor page faults during the loop: memory the kernel had to map on first touch.
    """
    if cpu is not None and warmup:
        raise ValueError("pool workers would inherit the pinned CPU; use cpu=None")
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})  # same as `taskset -c`, before any timed work
    files = sorted(p for p in IMAGES.glob("*/*") if p.is_file())
    sample = random.Random(seed).sample(files, k)[:n]
    if warmup:
        parallel_map(
            check_integrity,
            warmup_files(files, sample, warmup),
            workers=os.cpu_count() or 1,
            desc="Warm-up",
            unit="file",
        )
    for p in sample:  # warm the page cache
        p.read_bytes()
    cpus_seen = set()
    faults_before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    start = time.perf_counter()
    for i, p in enumerate(sample):
        check_integrity(p)
        if i % 500 == 0:
            cpus_seen.add(current_cpu())
    seconds = time.perf_counter() - start
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults_before
    return {
        **{"k": k, "n": n, "seed": seed, "cpu": cpu, "warmup": warmup},
        "ran_on": sorted(cpus_seen),
        "seconds": round(seconds, 2),
        "images_per_s": round(n / seconds),
        "faults_per_image": round(faults / n),
    }


def run_case(env: dict[str, str] | None = None, **options) -> dict:
    """Run one case in a new process; options override DEFAULTS (e.g. cpu=4, k=20000).

    env adds variables such as MALLOC_*, which glibc reads only at process start.
    """
    case = {**DEFAULTS, **options}
    out = subprocess.run(
        [sys.executable, __file__, json.dumps(case)],
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE,  # stderr not captured: a crash shows its traceback
        text=True,
        check=True,
    )
    result = json.loads(out.stdout)
    result["env"] = ";".join(f"{k}={v}" for k, v in (env or {}).items()) or "default"
    return result


def report(rows: list[dict], by: str | list[str], name: str) -> None:
    """Print every run and min/median/max per case; save the runs as a CSV.

    Saved to reports/experiment_<name>.csv, so numbers quoted in docs have a source.
    """
    import pandas as pd  # parent only, so the measured process stays minimal

    table = pd.DataFrame(rows).astype({"cpu": "Int64"})  # integer CPU, empty if None
    stats = table.groupby(by, dropna=False)[["images_per_s", "faults_per_image"]]
    # Full column widths: long labels such as the MALLOC_* settings stay readable.
    with pd.option_context("display.max_colwidth", None, "display.width", None):
        print(table.to_string(index=False))
        print(stats.agg(["min", "median", "max"]))
    out = REPORTS / f"experiment_{name}.csv"
    table.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # new process started by run_case: measure one case
        print(json.dumps(measure(**json.loads(sys.argv[1]))))
    else:
        # Cases alternate (P, E, unpinned, P, ...) so slow drift doesn't hit one only.
        rows = [
            run_case(**case) | {"repeat": r}
            for r in range(REPEATS)
            for case in CORE_TYPE_CASES
        ]
        report(rows, by="cpu", name="core_type")
