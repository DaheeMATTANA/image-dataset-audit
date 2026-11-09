"""Time the same 5,000 files serially, drawn with random.sample of different sizes.

Each case runs in a new process (run_case), with no warm-up pool. The timed files
are the first 5,000 of random.Random(42).sample(files, k), which are the same files
for k=5000 and k=20000: only the size of the list random.sample builds changes.
Each run records img/s and minor page faults per image. The runs are saved to
reports/experiment_sample_size.csv.

    uv run python experiments/time_sample_size_case.py
"""

from time_serial_case import REPEATS, report, run_case

SAMPLE_SIZES = [5000, 20000]

if __name__ == "__main__":
    # Cases alternate (5k, 20k, 5k, ...) so slow drift doesn't hit one case only.
    rows = [run_case(k=k) | {"repeat": r} for r in range(REPEATS) for k in SAMPLE_SIZES]
    report(rows, by="k", name="sample_size")
