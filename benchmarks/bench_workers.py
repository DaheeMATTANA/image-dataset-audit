"""Time check_integrity on image samples with different worker counts and chunk sizes.

Every file is read once before timing, so all runs start with a warm page cache
(cold reads are not measured). Each time includes starting the worker pool.
The 1-worker baseline runs once per sample size: one process ignores chunksize.
Output, one row per run: reports/benchmark_workers.csv when one chunksize is
timed, reports/benchmark_chunksize.csv when several are (so a quick run never
overwrites a chunksize comparison).
"""

import argparse
import os
import platform
import time
from pathlib import Path

import pandas as pd

from check_integrity import check_integrity
from image_dataset_audit.dataset import sample_images
from image_dataset_audit.parallel import START_METHOD, parallel_map

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SAMPLES = [5000]
# Worker counts worth trying on this machine (i7-13650HX: 6 P-cores + 8 E-cores,
# 20 logical CPUs): 6 = all P-cores, 14 = all physical cores, 20 = all logical CPUs
# including hyper-threads (os.cpu_count()).
DEFAULT_WORKERS = [os.cpu_count() or 1]
DEFAULT_CHUNKSIZES = [64]


def cpu_name() -> str:
    """CPU model from /proc/cpuinfo (Linux); falls back to the architecture name."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


def malloc_settings() -> str:
    """glibc malloc settings from the environment, or "default" if none are set.

    glibc reads them once at process start, so the environment of this process is
    what the timed runs used; worker processes inherit it.
    """
    names = sorted(k for k in os.environ if k.startswith("MALLOC_"))
    if "GLIBC_TUNABLES" in os.environ:
        names.append("GLIBC_TUNABLES")
    return ";".join(f"{k}={os.environ[k]}" for k in names) or "default"


def add_speedup(table: pd.DataFrame) -> pd.DataFrame:
    """Add speedup and efficiency against the 1-worker run of the same sample size."""
    baseline = table[table["workers"] == 1].set_index("sample")["seconds"]
    out = table.copy()
    out["speedup"] = out["sample"].map(baseline) / out["seconds"]
    out["efficiency"] = out["speedup"] / out["workers"]
    return out


def parse_args() -> argparse.Namespace:
    """Command-line options; exits with a usage error on invalid values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLES,
        help="sample sizes to time (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=DEFAULT_WORKERS,
        help="worker counts to time, e.g. 2 6 14 20; a 1-worker baseline is "
        "always added (default: %(default)s)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        nargs="+",
        default=DEFAULT_CHUNKSIZES,
        help="files sent to a worker at a time, e.g. 1 64 1024 (default: %(default)s)",
    )
    args = parser.parse_args()
    for name in ["sample", "workers", "chunksize"]:
        if min(getattr(args, name)) <= 0:
            parser.error(f"--{name} values must be positive integers")
    return args


def main() -> None:
    args = parse_args()
    workers = sorted(set(args.workers) - {1})
    chunksizes = sorted(set(args.chunksize))
    # (workers, chunksize) per run; None = not used by a single process.
    configs = [(1, None)] + [(w, c) for c in chunksizes for w in workers]
    machine = {
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "start_method": START_METHOD,
        "malloc": malloc_settings(),
    }
    print(" | ".join(f"{k}: {v}" for k, v in machine.items()))

    # One shuffled list; each sample is a prefix of it, so smaller samples are
    # subsets of larger ones and a single warm-up covers them all.
    order = sample_images(max(args.sample))
    sizes = sorted({min(n, len(order)) for n in args.sample})

    parallel_map(
        check_integrity,
        order,
        workers=max(w for w, _ in configs),
        desc="Warm-up",
        unit="file",
    )

    results = []
    for n in sizes:
        sample = order[:n]
        reference = None
        for w, c in configs:
            start = time.perf_counter()
            rows = parallel_map(
                check_integrity,
                sample,
                workers=w,
                chunksize=1 if c is None else c,  # ignored when w == 1
                desc=f"{n} files" if c is None else f"{n} files, chunksize {c}",
                unit="file",
            )
            seconds = time.perf_counter() - start

            df = pd.DataFrame(rows).sort_values("image_id", ignore_index=True)
            if reference is None:
                reference = df
            results.append(
                {
                    "sample": n,
                    "workers": w,
                    "chunksize": c,
                    "seconds": seconds,
                    "images_per_s": n / seconds,
                    "identical_output": df.equals(reference),
                    **machine,
                }
            )

    # Round only for output, after speedup is computed from the raw times.
    table = add_speedup(pd.DataFrame(results)).round(
        {"seconds": 2, "images_per_s": 1, "speedup": 2, "efficiency": 2}
    )
    # Integer column with empty cells for the 1-worker baseline.
    table["chunksize"] = table["chunksize"].astype("Int64")
    print(table.drop(columns=list(machine)).to_string(index=False))

    name = "chunksize" if len(chunksizes) > 1 else "workers"
    out = ROOT / "reports" / f"benchmark_{name}.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
