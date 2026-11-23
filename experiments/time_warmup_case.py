"""Time check_integrity in a serial loop after pool warm-ups of different sizes.

Each case runs in a new process (run_case): a pool with one worker per CPU decodes
`warmup` files, then the same 5,000 files are timed in a serial loop in that process.
The runs are saved to scratch/experiment_warmup.csv, or to reports/ with --save.

    uv run python experiments/time_warmup_case.py
"""

from time_serial_case import REPEATS, parse_save, report, run_case

WARMUPS = [0, 5000, 20000]

if __name__ == "__main__":
    save = parse_save(__doc__)
    # Cases alternate (0, 5k, 20k, 0, ...) so slow drift doesn't hit one case only.
    rows = [run_case(warmup=w) | {"repeat": r} for r in range(REPEATS) for w in WARMUPS]
    report(rows, by="warmup", name="warmup", save=save)
