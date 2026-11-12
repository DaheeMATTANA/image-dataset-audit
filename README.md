# image-dataset-audit

[![lint](https://github.com/DaheeMATTANA/image-dataset-audit/actions/workflows/lint.yml/badge.svg)](https://github.com/DaheeMATTANA/image-dataset-audit/actions/workflows/lint.yml)

A reproducible data-quality pipeline for image datasets. It checks dataset integrity, finds exact, near- and semantic duplicates, detects train/test leakage, and measures how much leakage changes a model's accuracy.

The first target is [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (101,000 images, 101 classes), a standard benchmark used from training tutorials to the evaluation suites of large vision-language models. The pipeline is designed to scale to datasets of ~10M images and, later, to video.

> **Status:** in progress. Stages 1 (dataset profile) and 2 (corruption and exact duplicates) are done; see [Results](#results-food-101). Stage 3 (near-duplicates) is next.

---

## Motivation

- **Duplicates waste compute and bias training:** the same image seen many times counts more than it should.
- **Train/test leakage inflates accuracy:** if a test image, or a resized copy of it, is also in the training set, the model is partly evaluated on memory rather than generalization.
- **Label noise and off-topic images:** an image can carry the wrong class or show no food at all; the pipeline flags them so they can be reviewed or removed.

---

## Pipeline

| # | Stage | Method | Output | Status |
|---|---|---|---|---|
| 1 | Dataset profile | Split and metadata checks; formats, color modes, resolutions, file sizes | Data profile | ✅ done |
| 2 | Corruption and exact duplicates | Full decode with Pillow, SHA-256 hashing | Corruption and exact-duplicate report | ✅ done |
| 3 | Near-duplicates | Perceptual hashing (dHash, pHash), Hamming distance, 200 hand-labeled pairs | Near-duplicate list with measured precision | ⚪ planned |
| 4 | Embeddings | Pretrained vision encoder (open_clip) | Embeddings for the full dataset | ⚪ planned |
| 5 | Semantic duplicates | Cosine similarity, threshold from a precision/recall curve | Semantic duplicates with precision and recall | ⚪ planned |
| 6 | Nearest-neighbor search at scale | Exact vs approximate search (FAISS) | Speed/recall trade-off table | ⚪ planned |
| 7 | Curation table | Columnar storage, one row per image, one column per signal | Parquet table | ⚪ planned |
| 8 | Leakage ablation | Linear probe on embeddings, original vs deduplicated split, 3 seeds | Accuracy difference with variance | ⚪ planned |
| 9 | Report and design doc | Constraints, options, decision, measured trade-offs | Audit report and design doc | ⚪ planned |

---

## Results: Food-101

Every number below is produced by a script in [`src/`](src/) or [`benchmarks/`](benchmarks/); full outputs are in [`reports/`](reports/).

### Stage 1: splits and metadata

Script: [`src/check_metadata.py`](src/check_metadata.py). Each row is one check run by the script. *Measured on* says what it reads: the split files (`meta/train.json`, `meta/test.json`), the image files on disk, or a comparison of the two.

| Check | Measured on | Observed |
|---|---|---|
| Classes | split files | 101 |
| Images per class (train / test) | split files | 750 / 250 in every class |
| Images on disk | disk | 101,000 |
| IDs listed twice within a split | split files | 0 |
| IDs in both train and test | split files | 0 |
| Listed in split files but missing on disk | split files vs disk | 0 |
| On disk but not listed in split files | split files vs disk | 0 |
| Class key ≠ class in the image ID | split files | 0 |
| File extensions | disk | `.jpg` (101,000) |

The set of image IDs in the split files matches the files on disk exactly (no missing, no unlisted files), so the per-class counts also hold on disk. The train/test check compares IDs only; resized or re-encoded copies across splits are covered by stage 3.

Reports: [`metadata_by_class.csv`](reports/metadata_by_class.csv) (per class), [`metadata_summary.csv`](reports/metadata_summary.csv) (one row per check).

### Stage 1: image profile

Script: [`src/profile_images.py`](src/profile_images.py) (reads image headers only, not pixels).

| Check | Observed |
|---|---|
| Readable image headers | 101,000 (0 unreadable) |
| Long side = 512 px | 100% |
| Short side | 122–512 px; 512 px (square) for 62%, 384 px for 21% |
| Orientation | 62,206 square, 26,219 landscape, 12,575 portrait |
| Format | JPEG (101,000) |
| Color mode | RGB (100,997), L / grayscale (3, all in train) |
| File size (1 KB = 1000 bytes) | median 49.6 KB, range 8.7–215.7 KB, 5.12 GB in total |
| File size, train vs test | near-identical distribution (median 49.6 KB train, 49.5 KB test; [plot](reports/profile_file_size.png)) |

The 3 grayscale images are not food photos (two show people, one shows text). They were checked by hand and are listed in [`annotations/flagged_images.csv`](annotations/flagged_images.csv). The color-mode check only finds images stored as grayscale; non-food images stored as RGB need later stages.

Report: [`profile_summary.csv`](reports/profile_summary.csv) (one row per check). Plots from [`src/plot_profile.py`](src/plot_profile.py): [short side](reports/profile_short_side.png), [width × height](reports/profile_width_height.png), [file size per split](reports/profile_file_size.png), [color mode](reports/profile_mode.png). The per-image table (`outputs/profile_images.csv`) is regenerated by the script and not committed.

### Stage 2: corruption

Script: [`src/check_integrity.py`](src/check_integrity.py). Every image is fully decoded with Pillow, which reads the whole file, not only the header.

| Check | Observed |
|---|---|
| Files checked | 101,000 (75,750 train, 25,250 test) |
| Decoded without error | 101,000 |
| Failed to decode | 0 |

Report: [`integrity_summary.csv`](reports/integrity_summary.csv) (also records the Pillow version). The per-image table (`outputs/integrity.csv`, with both hashes below) is regenerated by the script and not committed.

### Stage 2: exact duplicates

Script: [`src/find_exact_duplicates.py`](src/find_exact_duplicates.py), on the two SHA-256 hashes recorded by `check_integrity.py`:

- **File hash:** same value = byte-identical files.
- **Pixel hash:** same value = identical pixels after decoding (`convert("RGB")`, EXIF orientation not applied: what a typical training loader sees). It also finds copies whose files differ, for example in their metadata. Identical bytes always decode to identical pixels, so every file-hash group is part of a pixel-hash group.

A group is two or more images with the same hash. Each group is classified by split: within train, within test, or across train and test (leakage). *Extra copies* = images minus groups: the images a deduplication would remove, keeping one per group.

| Kind | Groups (file / pixel hash) | Images (file / pixel hash) | Extra copies (file / pixel hash) |
|---|---|---|---|
| Within train | 6 / 32 | 12 / 66 | 6 / 34 |
| Within test | 0 / 4 | 0 / 8 | 0 / 4 |
| Across train and test | 2 / 22 | 4 / 44 | 2 / 22 |
| **All** | **8 / 58** | **16 / 118** | **8 / 60** |

- **Extra copies:** 60 of the 101,000 images (0.059%) by pixel hash; 8 (0.008%) by file hash.
- **Leakage:** 22 of the 25,250 test images (0.087%) have an exact copy in train, by pixel hash.
- **Duplicates within test:** 8 test images (0.032%) are in 4 groups of copies inside the test set.
- **Label conflicts:** 2 groups (within train) hold the same image under two different classes; both are byte-identical copies.

All 58 pixel-hash groups were checked by hand; the verdicts are in [`annotations/exact_duplicate_review.csv`](annotations/exact_duplicate_review.csv) (columns described in [`annotations/README.md`](annotations/README.md)):

| Verdict | Groups | Kind |
|---|---|---|
| Same food photo, same class | 54 | 28 within train, 4 within test, 22 across train and test |
| Same photo under two classes | 2 | within train |
| Not a food photo (one shows people, one shows text) | 2 | within train |

The images with a wrong label (with the correct class) and the non-food images are also listed in [`annotations/flagged_images.csv`](annotations/flagged_images.csv). The non-food images are stored as RGB, so the color-mode check of stage 1 could not find them.

For the pixel-hash groups whose files are not byte-identical, the copies differ only in metadata (EXIF dates or GPS lines, Photoshop and XMP blocks, unused bytes); the encoded image data is byte-identical. See the [duplicate bytes experiment](experiments/README.md#why-pixel-identical-copies-differ-in-bytes).

Report: [`exact_duplicates_summary.csv`](reports/exact_duplicates_summary.csv) (one row per hash and kind, with percentages and label-conflict counts). The per-group tables (`outputs/exact_duplicates_file.csv`, `outputs/exact_duplicates_pixel.csv`) are regenerated by the script and not committed.

### Benchmark: integrity check throughput

Script: [`benchmarks/bench_workers.py`](benchmarks/bench_workers.py). It times `check_integrity` (stage 2) on a fixed-seed random sample of images against a 1-worker run: one run per configuration, warm page cache, worker pool start-up included, glibc `malloc` thresholds fixed (see [Benchmarks](#benchmarks)). Machine: Intel Core i7-13650HX (6 performance cores with 2 threads each, 8 efficiency cores; 20 logical CPUs), Python 3.12, `forkserver` start method.

Worker count, 20,000 images, chunk size 64:

| Workers | Images/s | Speedup | Efficiency |
|---|---|---|---|
| 1 | 691 | 1.00 | 1.00 |
| 2 | 1,307 | 1.89 | 0.95 |
| 4 | 2,616 | 3.79 | 0.95 |
| 6 | 3,737 | 5.41 | 0.90 |
| 14 | 5,160 | 7.47 | 0.53 |
| 20 | 5,276 | 7.63 | 0.38 |

Speedup is relative to the 1-worker run. Efficiency is speedup divided by the number of workers, so it counts every worker as a full performance core: beyond 6 workers, additional workers run on efficiency cores or share a performance core with another worker, so on this CPU efficiency also falls for hardware reasons, not only because of overhead.

Chunk size (files sent to a worker at a time), 5,000 images, images/s:

| Workers | 64 | 256 | 1024 |
|---|---|---|---|
| 2 | 1,238 | 1,252 | 1,109 |
| 6 | 3,097 | 2,811 | 2,738 |
| 20 | 3,693 | 3,434 | 2,545 |

With chunk size 1024, 5,000 images make only 5 batches, so at most 5 workers have work at a time. The 5,000-image runs at chunk size 64 in the two result files agree within 2%.

Plots: [worker count](reports/benchmark_workers_plot.png), [chunk size](reports/benchmark_chunksize_plot.png). Full results: [`benchmark_workers.csv`](reports/benchmark_workers.csv), [`benchmark_chunksize.csv`](reports/benchmark_chunksize.csv).

### Benchmark: threads vs processes

Script: [`benchmarks/bench_threads.py`](benchmarks/bench_threads.py). It runs the same `parallel_map(check_integrity, ...)` call with worker threads (one process) or worker processes, on the same fixed-seed sample of 5,000 images, chunk size 64. Each case runs in a new Python process; other conditions as above. With 1 worker, both run the same serial loop.

| Workers | Threads, images/s | Speedup | Processes, images/s | Speedup |
|---|---|---|---|---|
| 1 | 687 | 1.00 | 687 | 1.00 |
| 2 | 1,350 | 1.97 | 1,256 | 1.83 |
| 4 | 2,330 | 3.39 | 2,329 | 3.39 |
| 6 | 2,731 | 3.98 | 3,083 | 4.49 |
| 14 | 1,765 | 2.57 | 3,713 | 5.41 |
| 20 | 1,202 | 1.75 | 3,565 | 5.19 |

All threads in a process share one interpreter lock (the GIL): only the thread holding it runs Python code. In `check_integrity`, reading the file, hashing and JPEG decoding release the lock; the Python code around them (opening the file, building the result row) holds it. Up to 4 workers, threads match processes. Threads reach their highest throughput at 6 workers, which bounds the time each image holds the lock at about 1 / 2,731 s ≈ 0.37 ms (about a quarter of the 1.46 ms per image in the serial loop). Beyond 6 workers, throughput with threads falls, while processes, each with its own lock, keep scaling to 14 workers. Every run produces the same output as the serial loop (`identical_output`). For this CPU-bound stage the pipeline uses processes.

Plot: [threads vs processes](reports/benchmark_threads_plot.png). Full results: [`benchmark_threads.csv`](reports/benchmark_threads.csv).

---

## Getting started

### Requirements

- Linux or macOS
- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and all dependencies)
- ~10 GB of free disk space

### Installation

```bash
git clone https://github.com/DaheeMATTANA/image-dataset-audit.git
cd image-dataset-audit
uv sync
```

### Download the dataset

```bash
mkdir -p data
wget -P data http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf data/food-101.tar.gz -C data
```

The dataset is ~5 GB and lives in `data/`, which is git-ignored. Scripts treat it as read-only.

---

## Usage

Each stage is a standalone script. It reads from `data/`, writes small results (summaries, plots) to `reports/` and large per-image tables to `outputs/` (git-ignored), so every result can be regenerated.

| Script | Stage | Outputs |
|---|---|---|
| `src/check_metadata.py` | 1 | `reports/metadata_by_class.csv`, `reports/metadata_summary.csv` |
| `src/profile_images.py` | 1 | `outputs/profile_images.csv`, `reports/profile_summary.csv` |
| `src/plot_profile.py` | 1 | `reports/profile_*.png` (needs `outputs/profile_images.csv`) |
| `src/check_integrity.py` | 2 | `outputs/integrity.csv`, `reports/integrity_summary.csv` |
| `src/find_exact_duplicates.py` | 2 | `outputs/exact_duplicates_file.csv`, `outputs/exact_duplicates_pixel.csv`, `reports/exact_duplicates_summary.csv` (needs `outputs/integrity.csv`) |

```bash
uv run python src/check_metadata.py
uv run python src/profile_images.py
uv run python src/plot_profile.py
uv run python src/check_integrity.py --workers 0
uv run python src/find_exact_duplicates.py
```

`check_integrity.py` fully decodes every image and records two SHA-256 hashes per image: one of the file bytes, and one of the decoded pixels (after `convert("RGB")`, EXIF orientation not applied, which is what a typical training loader sees). Options:

- `--workers N`: number of worker processes (default 1; `0` = all CPUs). The output is the same for any number of workers.
- `--sample N`: check a random sample of N images (fixed seed); writes `outputs/integrity_sample.csv` only, no report.
- `--max-fail-rate F`: stop without writing the report if more than this fraction of files fail to decode (default 0.01), since a high failure rate more likely means a bug than a broken dataset.

### Benchmarks

Timing scripts live in [`benchmarks/`](benchmarks/) and write their results to `reports/benchmark_*`.

| Script | Measures | Outputs |
|---|---|---|
| `benchmarks/bench_workers.py` | `check_integrity` throughput for given sample sizes, worker counts and chunk sizes, against a 1-worker baseline (one run per configuration, warm page cache, pool start-up included; each row records the `malloc` settings used) | `reports/benchmark_workers.csv` (one chunk size) or `reports/benchmark_chunksize.csv` (several) |
| `benchmarks/plot_workers.py` | Throughput and efficiency per worker count, one line per sample size | `reports/benchmark_workers_plot.png` (needs `benchmark_workers.csv`) |
| `benchmarks/plot_chunksize.py` | Speedup per worker count, one line per chunk size, with the limit set by the number of batches (largest sample only) | `reports/benchmark_chunksize_plot.png` (needs `benchmark_chunksize.csv`) |
| `benchmarks/bench_threads.py` | `check_integrity` throughput with worker threads vs worker processes, against a serial loop (one new process per case, one run per case, warm page cache, pool start-up included) | `reports/benchmark_threads.csv` |
| `benchmarks/plot_threads.py` | Throughput per worker count, one line for threads and one for processes | `reports/benchmark_threads_plot.png` (needs `benchmark_threads.csv`) |

```bash
MALLOC="MALLOC_MMAP_THRESHOLD_=16777216 MALLOC_TRIM_THRESHOLD_=8388608"  # fixed malloc thresholds, see below
env $MALLOC uv run python benchmarks/bench_workers.py --sample 1000 5000 20000 --workers 2 4 6 14 20              # → benchmark_workers.csv
uv run python benchmarks/plot_workers.py
env $MALLOC uv run python benchmarks/bench_workers.py --sample 5000 --workers 2 4 6 14 20 --chunksize 64 256 1024  # → benchmark_chunksize.csv
uv run python benchmarks/plot_chunksize.py
env $MALLOC uv run python benchmarks/bench_threads.py                                                           # → benchmark_threads.csv
uv run python benchmarks/plot_threads.py
```

`env $MALLOC` sets the variables for that one command only, so nothing stays set in the shell afterwards (check with `env | grep MALLOC`).

**Memory allocator.** Decoding an image allocates a few large buffers. By default, glibc `malloc` adjusts its size thresholds to earlier allocations in the same process, so a process either reuses these buffers from one image to the next, or returns them to the operating system and maps fresh memory (page faults) for every image, which is slower. Which of the two a process gets depends on unrelated earlier work, such as drawing the random sample (tested in [`experiments/`](experiments/README.md#serial-throughput-and-the-memory-allocator)), and each worker process starts with its own state, so the 1-worker run and the workers can end up in different states. The two variables fix the thresholds, so every process reuses its buffers: mmap at 16.8 MB (above one decoded image of 0.8 MB, and within the 33.6 MB maximum documented for glibc) and trim at 8.4 MB (above the about 2.2 MB each image frees, while limiting the unused memory a process may keep). Both are needed; see the [threshold experiment](experiments/README.md#which-malloc-setting-removes-the-page-faults). The values are sized for Food-101's 512 px images: images with much larger decoded buffers need them measured again. glibc reads them when a process starts, so they are set on the command line, and worker processes inherit them; other C libraries (macOS, musl) ignore them. `src/check_integrity.py` itself runs with the default allocator, so its throughput can be lower than these benchmark numbers.

---

## Development

```bash
uv run ruff check .     # lint (add --fix to apply safe fixes)
uv run ruff format .    # format
uv run pytest           # tests
```

Ruff runs on every pull request through GitHub Actions ([`.github/workflows/lint.yml`](.github/workflows/lint.yml)). Changes go through pull requests to `master`.

### Project structure

```
image-dataset-audit/
├── .github/workflows/  # CI (lint)
├── src/                # one script per pipeline stage; shared helpers in src/image_dataset_audit/
├── experiments/        # one-off experiments (not pipeline stages), see experiments/README.md
├── benchmarks/         # timing scripts for pipeline stages (results in reports/benchmark_*)
├── tests/              # pytest tests
├── reports/            # small generated results: summaries, plots, the written report
├── docs/               # design docs: constraints, options, decisions, trade-offs
├── annotations/        # hand labels, keyed by image_id, see annotations/README.md
├── outputs/            # per-image tables, regenerated (git-ignored)
├── data/               # Food-101, local only (git-ignored, ~5 GB)
├── scratch/            # intermediate files from experiments (git-ignored, safe to delete)
├── pyproject.toml      # dependencies and tool config
└── uv.lock             # exact pinned versions
```

### Tools

Python 3.12 · uv · Pillow · pandas · NumPy · matplotlib · imagehash · tqdm · pytest · Ruff · PyTorch and open_clip · FAISS · pyarrow / Parquet

---

## Roadmap

- **Scale:** run the same pipeline on ~10M images across multiple machines, with throughput and cost measured.
- **Video:** extend the pipeline to video: decoding, scene detection, filtering, captioning and deduplication.

---

## Dataset and citation

Food-101 was created by Lukas Bossard, Matthieu Guillaumin and Luc Van Gool (ETH Zürich). The images come from foodspotting.com. This repository does not redistribute any images; see the [official dataset page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) for terms of use.

```bibtex
@inproceedings{bossard14,
  title     = {Food-101 -- Mining Discriminative Components with Random Forests},
  author    = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year      = {2014}
}
```
