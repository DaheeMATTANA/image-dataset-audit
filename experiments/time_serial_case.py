"""Time check_integrity in a serial loop, one case per new Python process.

Run directly to compare core types (CPU 4 = P-core, CPU 12 = E-core on this machine):

    uv run python experiments/time_serial_case.py

A new process per case means no case inherits another's memory state
(e.g. glibc malloc's). Writes nothing to the repo.
"""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from check_integrity import IMAGES, check_integrity

DEFAULTS = {"k": 5000, "n": 5000, "seed": 42, "cpu": None}
CORE_TYPE_CASES = [{"cpu": 4}, {"cpu": 12}, {}]


def current_cpu() -> int:
    """CPU this process last ran on (field 39 of /proc/self/stat, Linux)."""
    stat = Path("/proc/self/stat").read_text()
    return int(stat.rsplit(")", 1)[1].split()[36])


def measure(k: int, n: int, seed: int, cpu: int | None) -> dict:
    """Time check_integrity in a serial loop over n files.

    The files are the first n of random.Random(seed).sample(files, k).
    """
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})  # same as `taskset -c`, before any timed work
    files = sorted(p for p in IMAGES.glob("*/*") if p.is_file())
    sample = random.Random(seed).sample(files, k)[:n]
    for p in sample:  # warm the page cache
        p.read_bytes()
    cpus_seen = set()
    start = time.perf_counter()
    for i, p in enumerate(sample):
        check_integrity(p)
        if i % 500 == 0:
            cpus_seen.add(current_cpu())
    seconds = time.perf_counter() - start
    return {
        **{"k": k, "n": n, "seed": seed, "cpu": cpu},
        "ran_on": sorted(cpus_seen),
        "seconds": round(seconds, 2),
        "images_per_s": round(n / seconds),
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


if __name__ == "__main__":
    if len(sys.argv) > 1:  # new process started by run_case: measure one case
        print(json.dumps(measure(**json.loads(sys.argv[1]))))
    else:
        import pandas as pd  # parent only, so the measured process stays minimal

        rows = [run_case(**case) for case in CORE_TYPE_CASES]
        print(pd.DataFrame(rows).astype({"cpu": "Int64"}).to_string(index=False))
