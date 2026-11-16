# Experiments

One-off investigations, separate from the pipeline. Each one answers a single question about an image format, a hashing method or a performance result, on one image or a small sample. Experiments are not pipeline stages (those are in [`src/`](../src/)) and not benchmarks (those time pipeline stages and write `reports/benchmark_*`; see [`benchmarks/`](../benchmarks/)).

| Experiment | Question | Scripts | Output |
|---|---|---|---|
| [JPEG vs PNG](#jpeg-vs-png) | How do JPEG quality and repeated re-saving change file size and pixel values? | `jpeg_png_experiment.py` | `reports/jpeg_quality_iteration_experiment.png` |
| [Hashing and corruption checks](#hashing-and-corruption-checks) | Which changes do a file hash and a pixel hash detect, and which damaged files does each Pillow check catch? | `hashing_experiment.py` | Printed tables only |
| [Serial throughput and the memory allocator](#serial-throughput-and-the-memory-allocator) | Why does the same serial loop over the same files run at two different speeds? | `time_serial_case.py`, `time_warmup_case.py`, `time_sample_size_case.py`, `time_malloc_threshold_case.py` | `reports/experiment_*.csv` |
| [Why pixel-identical copies differ in bytes](#why-pixel-identical-copies-differ-in-bytes) | When two files decode to the same pixels but have different file hashes, what differs? | `duplicate_bytes_experiment.py` | `reports/experiment_duplicate_bytes.csv` |
| [Truncated hash collisions](#truncated-hash-collisions) | How many bits of a hash does an index need before different images share a value by chance? | `truncated_hash_experiment.py` | `reports/experiment_truncated_hash.csv` |
| [Hash index entry size](#hash-index-entry-size) | How much memory does one entry of an in-memory hash index take, and does a smaller entry format change the total much? | `index_entry_size_experiment.py` | `reports/experiment_index_entry_size.csv` |
| [dHash on edited copies](#dhash-on-edited-copies) | How far does a perceptual hash move when an image is resized, re-encoded, cropped or rotated, compared with unrelated images? | `dhash_experiment.py` | `reports/experiment_dhash.csv` |

Conventions:

- **Two kinds of scripts.** `jpeg_png_experiment.py`, `hashing_experiment.py` and `dhash_experiment.py` are exploration files with `# %%` cells, meant to be run cell by cell (VS Code, `ipykernel`). The `time_*_case.py` scripts are run whole from the terminal.
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

---

## Truncated hash collisions

**Question.** An index of hashes could store only the first bits of each SHA-256 to save memory. How many bits does it need before different images start sharing a value by chance, at Food-101's size and at 10M and 1B images?

**What it does.** Reads `file_sha256` and `pixel_sha256` from `outputs/integrity.csv` (stage 2), keeps the first N bits of each (N = 20, 24, 28, 32, 64, 256; one hex character holds 4 bits) and counts image pairs that share them. Pairs whose full 256-bit hashes also match are identical images; the others are chance collisions. Each count is compared with the expected number of chance collisions among n hashes of N bits:

```
expected = n (n − 1) / 2  ÷  2^N
         = number of pairs × chance that one pair of different images matches
```

Fewer bits on the same images give as many expected collisions as 32 bits on a larger dataset: at 101,000 images, 28 bits behave like 32 bits at about 400,000 images, 24 bits like about 1.6M, 20 bits like about 6.5M. These rows check the formula on large measured counts. At 10M and 1B images there is no data: the table gives the formula only, for 32, 64 and 256 bits.

**Observed** (101,000 images; chance pairs, expected in brackets):

| Bits | File hash | Pixel hash |
|---|---|---|
| 20 | 4,927 (4,864) | 4,833 (4,864) |
| 24 | 307 (304) | 309 (304) |
| 28 | 23 (19) | 23 (19) |
| 32 | 3 (1.19) | 4 (1.19) |
| 64 | 0 (1 in 3.6 billion) | 0 (1 in 3.6 billion) |
| 256 | 0 | 0 |

Identical pairs, at every width: 8 for the file hash, 62 for the pixel hash (the pixel-hash groups include copies that differ only in metadata bytes; see [Why pixel-identical copies differ in bytes](#why-pixel-identical-copies-differ-in-bytes)).

- At 20, 24 and 28 bits, every measured count is within one standard deviation of the expected value (√expected: about 70, 17 and 4).
- At 32 bits the counts are small (3 and 4 against 1.19), where a few pairs more or fewer than expected are common.

**Expected at larger sizes** (formula only, the same for both hashes; below 1, the value is about the chance of seeing even one chance pair):

| Images | 32 bits | 64 bits | 256 bits |
|---|---|---|---|
| 101,000 | 1.19 | 1 in 3.6 billion | < 1 in a trillion |
| 10M | 11,642 | 1 in 369,000 | < 1 in a trillion |
| 1B | 116 million | 1 in 37 | < 1 in a trillion |

The number of pairs grows with the square of the number of images, so 100× more images give about 10,000× more chance collisions at the same width. A 32-bit prefix already gives chance collisions at Food-101's size; 64 bits keep them unlikely at 10M but not at 1B; the full 256-bit hash keeps them negligible at every size shown.

Data: [`experiment_truncated_hash.csv`](../reports/experiment_truncated_hash.csv). One row per hash, width and dataset size. `pairs_sharing_bits` = `identical_pairs` + `chance_pairs`; these three are empty in the 10M and 1B rows. `expected_chance_pairs` is stored unrounded.

**Run** (reads `outputs/integrity.csv`, written by `src/check_integrity.py`; a few seconds):

```bash
uv run python experiments/truncated_hash_experiment.py
```

---

## Hash index entry size

**Question.** Finding exact duplicates needs an index from pixel hash to image: a hash already in the index is a duplicate. Kept in memory as a Python `dict`, how many bytes does one entry take, and how much do smaller key or value formats save?

**What it does.** Builds a `dict` from `pixel_sha256` to image for all images in `outputs/integrity.csv`, in 5 formats, and measures the bytes per entry with `sys.getsizeof`, in three parts:

- **Key:** the hash object. Hex string: the 64-character text stored in the CSV. 32-byte digest: the same hash as raw bytes. 8- and 4-byte prefixes: the first 64 and 32 bits of the digest.
- **Value:** the image. ID string: the image ID (e.g. `apple_pie/1005649`). Row number: the image's position in the main table, which is needed to look up the ID.
- **Slot:** the dict's own table, one row per entry pointing to the key and the value. `sys.getsizeof` on a dict counts only this table, so key and value objects are measured one by one.

Identical images share a key, so the dict holds one entry per unique hash: 100,940 for 101,000 images. The projections at 10M and 1B multiply the bytes per entry by the number of images.

**Observed** (bytes per entry):

| Key | Value | Entries | Key | Value | Slot | Total | At 10M | At 1B |
|---|---|---|---|---|---|---|---|---|
| hex string | ID string | 100,940 | 105.0 | 59.4 | 38.1 | 202.5 | 2.02 GB | 202.5 GB |
| 32-byte digest | ID string | 100,940 | 65.0 | 59.4 | 51.9 | 176.3 | 1.76 GB | 176.3 GB |
| 8-byte prefix | ID string | 100,940 | 41.0 | 59.4 | 51.9 | 152.3 | 1.52 GB | 152.3 GB |
| 4-byte prefix | ID string | 100,936 | 37.0 | 59.4 | 51.9 | 148.3 | 1.48 GB | 148.3 GB |
| 32-byte digest | row number | 100,940 | 65.0 | 28.0 | 51.9 | 144.9 | 1.45 GB | 144.9 GB |

- **Most of an entry is Python overhead, not data.** Every object carries a fixed header of about 33 to 41 bytes whatever its content: a 32-byte digest takes 65 bytes, a 4-byte prefix 37, a 17-character ID 58.
- **Smaller formats change the total by at most 18%** against the full digest with the ID string (176.3 → 144.9 bytes, with a row number as value). Truncating the key to 4 bytes saves 28 bytes per entry and merges different images: 100,936 entries instead of 100,940, the 4 chance pairs of the pixel hash at 32 bits (see [Truncated hash collisions](#truncated-hash-collisions)).
- **Text keys use smaller table rows** (38.1 vs 51.9 bytes). A `str` object stores its own hash, so a dict whose keys are all strings keeps two pointers per row (16 bytes) instead of the hash and two pointers (24 bytes). The 40 extra bytes of the hex string itself outweigh the gain.
- **At 10M, every format fits in memory on one machine** (1.45–2.02 GB). At 1B, none does (145–203 GB).

The projections are approximate. The dict reserves its table in steps (powers of two), so slot bytes per entry change with the number of entries, and they assume IDs as long as Food-101's.

Data: [`experiment_index_entry_size.csv`](../reports/experiment_index_entry_size.csv). One row per format. Byte columns are averages per entry; `gb_at_10m` and `gb_at_1b` are in GB (10⁹ bytes).

**Run** (reads `outputs/integrity.csv`, written by `src/check_integrity.py`; a few seconds):

```bash
uv run python experiments/index_entry_size_experiment.py
```

---

## dHash on edited copies

**Question.** A perceptual hash should stay close when an image is edited but still shows the same photo, and be far for a different photo. How far does a 64-bit dHash move for common edits, compared with unrelated images?

**What it does.** dHash converts an image to grayscale, resizes it to 9 × 8 pixels (Lanczos filter) and compares each pixel with its left neighbor: 8 rows × 8 comparisons give 64 bits, 1 where the right pixel is brighter. The Hamming distance between two hashes is the number of bits that differ: 0 for the same hash, 32 on average for two random 64-bit values.

The script computes dHash with NumPy and checks it against `imagehash.dhash`: same 64 bits on all 7 images, in all 4 rotations. It then compares one image (`apple_pie/1005649`) with four edited versions of it (resized from 512 to 256 px on the long side, re-saved as JPEG at quality 50, cropped by 10% on each side, rotated 90°), another image of the same class (`apple_pie/1011328`) and an image of another class (`pizza/1001116`). Each comparison is also made rotation-aware: the second image is hashed at 0°, 90°, 180° and 270°, and the smallest distance is kept.

**Observed** (Hamming distance to the original, out of 64 bits):

| Compared image | Distance | Best of 4 rotations |
|---|---|---|
| Resized to 256 px | 0 | 0 |
| Re-saved at JPEG quality 50 | 0 | 0 |
| Cropped 10% per side | 14 | 14 |
| Rotated 90° | 33 | 0 |
| Same class, other image | 27 | 27 |
| Other class | 41 | 31 |

- **Resizing and re-encoding change no bit.** Each of the 9 × 8 pixels averages a block of thousands of original pixels; these edits move the averages too little to change which neighbor is brighter.
- **Cropping changes 14 bits.** Pixel values stay the same, but each of the 9 × 8 cells now covers a different part of the photo.
- **A 90° rotation gives 33, as far as the unrelated images.** dHash compares left and right neighbors; after rotation, these are different pairs of cells. Hashing all 4 rotations brings it to 0: rotating in 90° steps only moves pixels, so turning the copy back gives the original pixels.
- **Rotation-aware matching also lowers unrelated pairs:** the other-class image goes from 41 to 31. The smallest of 4 distances is never larger than one of them, so rotated copies are caught at the cost of more unrelated pairs falling under a given threshold.
- One image per case: the distances show how each edit acts on dHash, not how often each edit occurs in Food-101. No near-duplicate threshold is chosen here.

Data: [`experiment_dhash.csv`](../reports/experiment_dhash.csv). One row per compared image; `distance` is to the original as is, `best_of_4_rotations` the smallest distance over the 4 rotations of the compared image.

**Run** (reads 3 images from `data/`; about a second). Open `experiments/dhash_experiment.py` in VS Code and run the cells in order, or:

```bash
uv run python experiments/dhash_experiment.py
```
