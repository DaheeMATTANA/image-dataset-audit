# Experiments

One-off investigations, separate from the pipeline. Each one answers a single question about an image format, a hashing method or a performance result, on one image or a small sample. Experiments are not pipeline stages (those are in [`src/`](../src/)) and not benchmarks (those time pipeline stages and write `reports/benchmark_*`; see [`benchmarks/`](../benchmarks/)).

| Experiment | Question | Scripts | Output |
|---|---|---|---|
| [JPEG vs PNG](#jpeg-vs-png) | How do JPEG quality and repeated re-saving change file size and pixel values? | `jpeg_png_experiment.py` | `reports/jpeg_quality_iteration_experiment.png` |
| [Hashing and corruption checks](#hashing-and-corruption-checks) | Which changes do a file hash and a pixel hash detect, and which damaged files does each Pillow check catch? | `hashing_experiment.py` | Printed tables only |
| [Serial throughput and the memory allocator](#serial-throughput-and-the-memory-allocator) | Why does the same serial loop over the same files run at two different speeds? | `time_serial_case.py`, `time_warmup_case.py`, `time_sample_size_case.py`, `time_malloc_threshold_case.py` | `reports/experiment_*.csv` |
| [Why pixel-identical copies differ in bytes](#why-pixel-identical-copies-differ-in-bytes) | When two files decode to the same pixels but have different file hashes, what differs? | `duplicate_bytes_experiment.py` | `reports/experiment_duplicate_bytes.csv` |

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

**Setup** (shared by the four scripts, in `time_serial_case.py`):

- The timed loop runs `check_integrity` (stage 2: full decode and two hashes) serially on 5,000 files: the first 5,000 of `random.Random(42).sample(files, k)`. With seed 42, `k=5000` and `k=20000` give the same first 5,000 files.
- Every file is read once before timing, so the page cache is warm.
- Each case runs in a new Python process with the default glibc `malloc` settings (no `MALLOC_*` variables), except in the `malloc` threshold cases.
- Each case runs 3 times; cases alternate (A, B, C, A, B, C, …) so a slow drift does not affect one case only. The `malloc` threshold cases run once each: their page-fault counts are exact, and the gaps between cases are far larger than run-to-run noise.
- Machine: Intel Core i7-13650HX. CPUs 0–11 are the 6 performance cores (P-cores, 2 threads each, up to 4.7–4.9 GHz); CPUs 12–19 are the 8 efficiency cores (E-cores, up to 3.6 GHz). Python 3.12, AC power, no other programs running.

**Results** (img/s, median of 3 runs, min–max in brackets):

| Possible cause | Script | Cases | img/s | Result |
|---|---|---|---|---|
| Core type: the slow runs ran on efficiency cores | `time_serial_case.py` | Pinned to CPU 4 (P-core) / CPU 12 (E-core) / not pinned; `k=5000` | 529 (528–532) / 345 (345–359) / 509 (509–532) | Not the cause. E-cores are slower, but on a P-core the loop stays far below the fast speed (707) |
| Warm-up size: the 20,000-file run decoded more files first | `time_warmup_case.py` | Pool warm-up of 0 / 5,000 / 20,000 files in the same process, before the timed loop; `k=5000` | 530 (513–531) / 523 (510–530) / 507 (505–512) | Not the cause: all three stay at the slow speed; the 20,000-file warm-up is slightly slower, not faster |
| Sample size: the size of the list `random.sample` builds | `time_sample_size_case.py` | `k=5000` / `k=20000`, same 5,000 timed files | 511 (511–531) / 704 (673–706); page faults per image: 526 / 0 | The cause: only `k` changes; the loop is 1.4× faster and has no page faults |

**Page faults.** A minor page fault happens when a process first touches memory the kernel has not mapped to RAM yet. With `k=5000`, every run has 526 faults per image: about 2.2 MB (526 × 4 KB pages) of memory mapped again for each image. With `k=20000`, there are none. Counted with `resource.getrusage().ru_minflt` before and after the timed loop.

### Which `malloc` setting removes the page faults

glibc `malloc` uses two size thresholds:

- **mmap threshold:** a request larger than this gets its own block of memory from the kernel, returned to the kernel when freed.
- **Trim threshold:** when more free memory than this piles up at the end of the heap (the area `malloc` reuses), it is returned to the kernel.

Memory returned to the kernel has to be mapped again, page by page, the next time it is used. By default glibc adjusts both thresholds to the earlier allocations of the process. Setting either one with a `MALLOC_*` variable switches that adjustment off for both (glibc `mallopt(3)`).

`time_malloc_threshold_case.py` runs the slow case (`k=5000`) with fixed thresholds, one run per case ("default" = not set, glibc adjusts it):

| mmap threshold | Trim threshold | img/s | Page faults per image | Reading |
|---|---|---|---|---|
| default | default | 508 | 526 | The slow case |
| 16.8 MB | default | 501 | 593 | mmap alone: worse than default |
| default | 134.2 MB | 654 | 123 | Trim alone: most faults removed, not all |
| 16.8 MB | 134.2 MB | 676 | 0 | Both: no faults |
| 67.1 MB | 134.2 MB | 709 | 0 | Both, mmap above the documented maximum |
| 16.8 MB | 1.0 MB | 503 | 593 | Trim below what one image frees |
| 16.8 MB | 2.1 MB | 529 | 527 | Trim just below what one image frees |
| 16.8 MB | 4.2 MB | 700 | 0 | Smallest tested trim value with no faults |
| 16.8 MB | 8.4 MB | 675 | 0 | The values used in the [benchmark commands](../README.md#benchmarks) |

- **Both thresholds are needed.** The rows with only one threshold set are tests, not usable settings: neither removes the faults on its own.
- **The trim threshold must be larger than what one image frees** (about 2.2 MB): 2.1 MB still gives a fault count close to default; 4.2 MB and above give none. Larger values add no speed, but each process may keep up to that much unused memory instead of returning it.
- **glibc 2.43 accepts an mmap threshold of 67.1 MB,** above the 33.6 MB maximum given in the `mallopt(3)` documentation: with it, the faults disappear, as with 16.8 MB.
- With one run per case, img/s of the no-fault cases varies from 675 to 709; compare cases by page faults, which are exact counts.

**Not shown yet.** How the sample size `k` sets the thresholds in the default case, before any image is decoded.

Data: [`experiment_core_type.csv`](../reports/experiment_core_type.csv), [`experiment_warmup.csv`](../reports/experiment_warmup.csv), [`experiment_sample_size.csv`](../reports/experiment_sample_size.csv), [`experiment_malloc_threshold.csv`](../reports/experiment_malloc_threshold.csv). One row per run. `cpu` is the CPU the process was pinned to (empty if not pinned), `ran_on` the CPUs it was seen on during the loop, `env` the extra environment variables (`default` if none), `faults_per_image` the minor page faults during the loop divided by the number of images (sample-size and threshold runs), `mmap_threshold_mb` and `trim_threshold_mb` the thresholds set, in MB (empty if not set).

**Run** (from the repo root, on AC power with nothing else running, about 2 minutes each; no `MALLOC_*` variable exported in the shell):

```bash
uv run python experiments/time_serial_case.py            # core type → reports/experiment_core_type.csv
uv run python experiments/time_warmup_case.py            # warm-up size → reports/experiment_warmup.csv
uv run python experiments/time_sample_size_case.py       # sample size → reports/experiment_sample_size.csv
uv run python experiments/time_malloc_threshold_case.py  # malloc thresholds → reports/experiment_malloc_threshold.csv
```

---

## Why pixel-identical copies differ in bytes

**Question.** Stage 2 finds groups of images with the same pixel hash. In some groups the files are not byte-identical (different file hashes). What differs between those files, if the decoded pixels are the same?

**What it does.** Takes the pixel-hash groups with more than one file hash (from `annotations/exact_duplicate_review.csv`) and compares each copy with its group's first image. Each file is read and parsed once. A JPEG file is a sequence of blocks: metadata blocks (EXIF, XMP, Photoshop, ICC, comments), which the decoder skips, and decoder blocks (quantization tables, Huffman tables, frame header, image data). For each pair:

- **Category:** `identical`; `metadata` (only metadata blocks differ); `coding` (same quantization tables, other decoder blocks differ: likely the same image compressed another way); `quantization` (quantization tables differ: likely saved again). The categories describe what differs, not the file's history.
- **Reason:** which metadata blocks differ and, inside EXIF, which tags differ. When EXIF bytes differ but no tag value does, and the differing bytes are zero in one copy, the reason is `unused bytes`.
- **Flags:** different file sizes; orientation tag differs as stored; the copies would display rotated differently in software that applies the tag (missing or invalid values count as upright).

**Observed** (50 groups, 52 compared pairs):

| Category | Pairs |
|---|---|
| identical | 0 |
| metadata | 52 |
| coding | 0 |
| quantization | 0 |

In every pair the decoder blocks, including the encoded image data, are byte-identical: the copies differ only in metadata.

| Reason | Pairs |
|---|---|
| EXIF date tags only (`DateTime`, `DateTimeDigitized`, `DateTimeOriginal`) | 43 |
| EXIF GPS position tags (with or without a date tag) | 6 |
| EXIF date tag and GPS method text | 1 |
| Photoshop and XMP blocks (EXIF identical) | 1 |
| EXIF unused bytes (no tag value differs) | 1 |

- In the 50 pairs of same-size files, 2 to 39 bytes differ (median 5).
- File sizes differ in 2 pairs, in both because a metadata block is longer in one copy.
- The orientation tag differs in 1 pair (present in one copy, missing in the other), but both values mean upright: 0 pairs would display rotated differently.
- Some files have damaged EXIF tags, which Pillow skips; a difference inside a skipped tag cannot be named. The category is not affected: it compares bytes.

The report holds tag and block names only, no metadata values (dates, GPS positions).

**Run** (reads `annotations/exact_duplicate_review.csv` and the images in `data/`):

```bash
uv run python experiments/duplicate_bytes_experiment.py
```
