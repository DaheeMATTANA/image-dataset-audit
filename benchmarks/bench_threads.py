"""Time check_integrity with a pool of threads vs a pool of processes.

Each case runs in a new Python process, so no case inherits another's memory
state (e.g. glibc malloc's). Both kinds call parallel_map with the same
chunksize; only the kind of worker differs. With 1 worker parallel_map runs a
plain loop, the baseline for speedup. Every file is read once before timing
(warm page cache); each time includes starting the pool. One run per case.
Output, one row per case: reports/benchmark_threads.csv
"""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from bench_workers import cpu_name, malloc_settings
from check_integrity import check_integrity
from image_dataset_audit.dataset import IMAGES, REPORTS
from image_dataset_audit.parallel import START_METHOD, parallel_map

SEED = 42
# 6 = all P-cores, 14 = all physical cores, 20 = all logical CPUs (i7-13650HX);
# same set as reports/benchmark_workers.csv, so process rows can be compared.
DEFAULT_WORKERS = [1, 2, 4, 6, 14, 20]
KINDS = ["thread", "process"]


def sample_files(n: int) -> list[Path]:
    """The same fixed-seed random sample of n files, in every process."""
    files = sorted(p for p in IMAGES.glob("*/*") if p.is_file())
    if not files:
        sys.exit(f"No files found under {IMAGES}")
    return random.Random(SEED).sample(files, min(n, len(files)))


def measure(kind: str, workers: int, n: int) -> dict:
    """Time parallel_map(check_integrity) over the sample in this process."""
    sample = sample_files(n)
    start = time.perf_counter()
    rows = parallel_map(
        check_integrity, sample, workers=workers, threads=(kind == "thread")
    )
    seconds = time.perf_counter() - start
    # Fingerprint of the sorted output, so the parent can check all cases agree.
    rows.sort(key=lambda r: r["image_id"])
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    return {
        "kind": kind,
        "workers": workers,
        "sample": len(sample),
        "chunksize": None if kind == "serial" else 64,  # parallel_map's default
        "seconds": seconds,
        "output_sha256": digest,
    }


def run_case(case: dict) -> dict:
    """Run measure(**case) in a new Python process.

    The new process inherits this environment, MALLOC_* included.
    """
    out = subprocess.run(
        [sys.executable, __file__, "--case", json.dumps(case)],
        stdout=subprocess.PIPE,  # stderr not captured: a crash shows its traceback
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def parse_args() -> argparse.Namespace:
    """Command-line options; exits with a usage error on invalid values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="files to time (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=DEFAULT_WORKERS,
        help="worker counts for each pool (default: %(default)s)",
    )
    parser.add_argument("--case", help=argparse.SUPPRESS)  # set by run_case only
    args = parser.parse_args()
    if args.sample <= 0 or min(args.workers) <= 0:
        parser.error("--sample and --workers values must be positive integers")
    return args


def main() -> None:
    args = parse_args()
    if args.case:  # new process started by run_case: time one case, print JSON
        print(json.dumps(measure(**json.loads(args.case))))
        return

    machine = {
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "start_method": START_METHOD,
        "malloc": malloc_settings(),
    }
    print(" | ".join(f"{k}: {v}" for k, v in machine.items()))

    for p in sample_files(args.sample):  # warm the page cache (shared by processes)
        p.read_bytes()

    cases = [{"kind": "serial", "workers": 1, "n": args.sample}] + [
        {"kind": kind, "workers": w, "n": args.sample}
        for w in sorted(set(args.workers) - {1})  # 1 worker = the serial loop
        for kind in KINDS
    ]
    table = pd.DataFrame([run_case(c) for c in tqdm(cases, desc="Cases")])

    serial = table.loc[table["kind"] == "serial", "seconds"].item()
    table["images_per_s"] = table["sample"] / table["seconds"]
    table["speedup"] = serial / table["seconds"]
    table["identical_output"] = table["output_sha256"] == table["output_sha256"][0]
    table["chunksize"] = table["chunksize"].astype("Int64")  # empty for serial
    table = table.drop(columns="output_sha256").assign(**machine)
    table = table.round({"seconds": 2, "images_per_s": 1, "speedup": 2})

    print(table.drop(columns=list(machine)).to_string(index=False))
    print(table.pivot(index="workers", columns="kind", values="images_per_s"))

    out = REPORTS / "benchmark_threads.csv"
    table.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
