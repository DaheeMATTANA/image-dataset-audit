# Experiments

One-off investigations, separate from the pipeline. Each one answers a single question about an image format, a hashing method or a performance result, on one image or a small sample. Experiments are not pipeline stages (those are in [`src/`](../src/)) and not benchmarks (those time pipeline stages and write `reports/benchmark_*`; see [`benchmarks/`](../benchmarks/)).

| Experiment | Question | Scripts | Output |
|---|---|---|---|
| [JPEG vs PNG](#jpeg-vs-png) | How do JPEG quality and repeated re-saving change file size and pixel values? | `jpeg_png_experiment.py` | `reports/jpeg_quality_iteration_experiment.png` |
| [Hashing and corruption checks](#hashing-and-corruption-checks) | Which changes do a file hash and a pixel hash detect, and which damaged files does each Pillow check catch? | `hashing_experiment.py` | Printed tables only |
| [Serial throughput and the memory allocator](#serial-throughput-and-the-memory-allocator) | Why does the same serial loop over the same files run at two different speeds? | `time_serial_case.py`, `time_warmup_case.py`, `time_sample_size_case.py` | `reports/experiment_*.csv` |

Conventions:

- **Two kinds of scripts.** `jpeg_png_experiment.py` and `hashing_experiment.py` are exploration files with `# %%` cells, meant to be run cell by cell (VS Code, `ipykernel`). The `time_*_case.py` scripts are run whole from the terminal.
- **Timed cases run in a new process,** never in an interactive kernel: earlier work in the same process (for example the state of glibc `malloc`) changes the speed.
- **Run timings on AC power, with nothing else running.** On battery the laptop lowers CPU clock speeds, and other programs compete for cores and for the CPU's power budget; either makes cases slower and less stable.
- **Outputs.** Intermediate files go to `scratch/` (git-ignored). Results quoted in this file are saved in `reports/`.

---

## JPEG vs PNG

**Question.** How much do JPEG quality and repeated re-saving change file size and pixel values, compared with lossless PNG?

**What it does.** Takes one Food-101 image, saves it as JPEG at quality 95, 50 and 10 and as PNG, and measures file size and mean absolute pixel error against the original pixels. It then re-saves the image 10 times in a row at a fixed quality (95 and 80), each time from the previous copy.

**Observed** ([plot](../reports/jpeg_quality_iteration_experiment.png)):

- At quality 80, error and file size are the same at every generation: after the first save, re-saving gives back the same pixels.
- At quality 95, error rises over the generations and levels off near generation 10, while file size stays almost constant.
- PNG has zero pixel error: it is lossless.

**Run.** Open `experiments/jpeg_png_experiment.py` in VS Code and run the cells in order. Intermediate files go to `scratch/toy_jpeg/`.

---

## Hashing and corruption checks

**Question.** What does a SHA-256 hash of the file detect, compared with a hash of the decoded pixels? Which damaged files does each Pillow check (`open`, `verify`, `load`) catch?

**What it does.**

1. **File hash.** Copies an image (same hash), then flips one bit and counts how many of the 256 output bits change, over 1,000 random flips (avalanche effect).
2. **File hash vs pixel hash.** Compares an exact copy, a copy with extra bytes appended, and a copy re-saved at quality 95. The pixel hash is the SHA-256 of the RGB pixels after decoding, the same pixels a training loader sees.
3. **Corruption checks.** Builds damaged copies (truncated to 50% and 99%, empty file, 1,000 random bytes mid-file, a PNG named `.jpg`) and runs each Pillow check level on each.

This is the basis for `src/check_integrity.py`, which records both hashes and runs a full decode (`load`) on every image.

**Run.** Open `experiments/hashing_experiment.py` in VS Code and run the cells in order. Damaged copies go to `scratch/hashing/` and `scratch/bad_copies/`.

---

## Serial throughput and the memory allocator

**Observation.** Before the allocator settings were fixed for benchmarks, the 1-worker baseline in `benchmarks/bench_workers.py` ran at different speeds depending on the sample sizes requested: slower with `--sample 5000` than with `--sample 1000 5000 20000`, on the same first 5,000 files. The two runs differed in three ways, each a possible cause, and each tested here with one script that changes only that one thing.

**Setup** (shared by the three scripts, in `time_serial_case.py`):

- The timed loop runs `check_integrity` (stage 2: full decode and two hashes) serially on 5,000 files: the first 5,000 of `random.Random(42).sample(files, k)`. With seed 42, `k=5000` and `k=20000` give the same first 5,000 files.
- Every file is read once before timing, so the page cache is warm.
- Each case runs in a new Python process with the default glibc `malloc` settings (no `MALLOC_*` variables).
- Each case runs 3 times; cases alternate (A, B, C, A, B, C, …) so a slow drift does not affect one case only.
- Machine: Intel Core i7-13650HX. CPUs 0–11 are the 6 performance cores (P-cores, 2 threads each, up to 4.7–4.9 GHz); CPUs 12–19 are the 8 efficiency cores (E-cores, up to 3.6 GHz). Python 3.12, AC power, no other programs running.

**Results** (img/s, median of 3 runs, min–max in brackets):

| Possible cause | Script | Cases | img/s | Result |
|---|---|---|---|---|
| Core type: the slow runs ran on efficiency cores | `time_serial_case.py` | Pinned to CPU 4 (P-core) / CPU 12 (E-core) / not pinned; `k=5000` | 529 (528–532) / 345 (345–359) / 509 (509–532) | Not the cause. E-cores are slower, but on a P-core the loop stays far below the fast speed (707) |
| Warm-up size: the 20,000-file run decoded more files first | `time_warmup_case.py` | Pool warm-up of 0 / 5,000 / 20,000 files in the same process, before the timed loop; `k=5000` | 530 (513–531) / 523 (510–530) / 507 (505–512) | Not the cause: all three stay at the slow speed; the 20,000-file warm-up is slightly slower, not faster |
| Sample size: the size of the list `random.sample` builds | `time_sample_size_case.py` | `k=5000` / `k=20000`, same 5,000 timed files | 528 (511–531) / 707 (677–712) | The cause: only `k` changes, and the loop is 1.3× faster |

Data: [`experiment_core_type.csv`](../reports/experiment_core_type.csv), [`experiment_warmup.csv`](../reports/experiment_warmup.csv), [`experiment_sample_size.csv`](../reports/experiment_sample_size.csv). One row per run. `cpu` is the CPU the process was pinned to (empty if not pinned), `ran_on` the CPUs it was seen on during the loop, `env` the extra environment variables (`default` if none).

**Not shown yet.** How the size of a Python list built before the loop changes the speed of decoding images later. The main [README](../README.md#benchmarks) explains it through glibc `malloc` thresholds and page faults; the page-fault count per image is the next measurement to add here.

**Run** (from the repo root, on AC power with nothing else running, about 2 minutes each):

```bash
uv run python experiments/time_serial_case.py        # core type → reports/experiment_core_type.csv
uv run python experiments/time_warmup_case.py        # warm-up size → reports/experiment_warmup.csv
uv run python experiments/time_sample_size_case.py   # sample size → reports/experiment_sample_size.csv
```
