# Design doc: Phase 1, Food-101 audit

*Status: draft v0. Sections 3–5 are completed as each pipeline stage is implemented.*

*Source labels:*
*- **[measured]**: written by a script in `src/` to `reports/`.*
*- **[per-image table]**: computed from `outputs/profile_images.csv`, not yet in a report.*
*- **[hand-labeled]**: human judgement recorded in `annotations/`.*
*- **[timed manually]**: one timed run, not yet recorded by a script.*
*- **[estimate]**: back-of-envelope, checked against measurements in section 5.*

---

## 1. Problem and constraints

**Problem.** Food-101 is a widely used image classification benchmark. Before a dataset is used for training or evaluation, a curation pipeline needs a precise, file-level picture of its contents. This phase builds that picture by auditing the dataset for:

- corrupted or unreadable files
- exact duplicates (byte-identical)
- near-duplicates (same photo resized or re-encoded)
- semantic duplicates (different photos of the same thing)
- train/test leakage (any of the above crossing the official split)

and then quantifies **the effect of cross-split duplicates on evaluation accuracy** (linear probe on image embeddings, original split vs deduplicated split, 3 seeds).

*Open question: the deduplicated test set is a different set of images from the original one, so an accuracy gap could come from leakage or from the remaining images being easier or harder. The control that separates the two is still to be chosen.*

**Data** (from `reports/metadata_summary.csv` and `reports/profile_summary.csv` unless stated):

| Property | Value | Source |
|---|---|---|
| Images | 101,000 files, 101 classes | [measured] |
| Official split | 750 train / 250 test per class (75,750 / 25,250), no ID in both | [measured] |
| Format | 100% JPEG | [measured] |
| Resolution | long side = 512 px for 100% of images; short side 122–512 px | [measured] |
| Square images (512 × 512) | 62% | [per-image table] |
| Color mode | 100,997 RGB, 3 grayscale (all train) | [measured] |
| Grayscale images | none of the 3 shows food | [hand-labeled] `annotations/flagged_images.csv` |
| Size on disk | 5.12 GB in total; median 49.6 KB, mean 50.7 KB per image (1 KB = 1,000 bytes) | [measured] |

**Constraints.**

- **Hardware:** one laptop, CPU only: Intel Core i7-13650HX (14 cores, 20 threads), 30 GB RAM. Embedding extraction runs on an on-demand cloud GPU.
- **Budget:** minimal, amount still to set. Estimated GPU cost: under ~$2 per embedding run (section 2).
- **Milestone:** audit report and this design doc v1 at the end of Phase 1.
- **Data is read-only:** nothing in `data/` is modified; decisions are recorded in a table, never applied by deleting files.
- **Reproducibility:** every number comes from a script that runs end to end; all per-image results share the key `image_id`.

---

## 2. Back-of-envelope estimates

### Storage and memory

| Quantity | Calculation | Result |
|---|---|---|
| Mean file size | 5.12 GB ÷ 101,000 | 50.7 KB (median 49.6 KB) [measured] |
| One decoded image in memory | 512 × 512 × 3 bytes (square) down to 512 × 384 × 3 | 0.59–0.79 MB; mean 0.705 MB [per-image table] |
| Whole dataset decoded | sum of H × W × 3 over all images | **71.2 GB** [per-image table], vs 30 GB of RAM: every pass must **stream** images, never load them all |
| Compression ratio of the JPEGs | 0.705 MB decoded ÷ 50.7 KB on disk | ~14× |
| Embeddings (512 dims, float32) | 101,000 × 512 × 4 bytes | ~207 MB (~103 MB in float16): fits in RAM. 512 dims assumes a ViT-B-size encoder (e.g. CLIP ViT-B/32); a ViT-L encoder outputs 768 dims (~310 MB) |
| Curation table | 101,000 rows × ~20 columns | a few MB as Parquet |

### Time per pass

| Pass | Cost per image | Whole dataset | Status |
|---|---|---|---|
| Header profile (format, size, mode) | 0.14 ms cold cache, ~0.05 ms warm | **14 s** cold (first run), **~5 s** warm (repeat), 1 process | [timed manually] |
| Full decode (corruption check) | ~3–5 ms | ~5–8 min with 1 process; ~1 min with 14 processes | [estimate] |
| SHA-256 of every file | cold: limited by disk reads; warm (5.12 GB fits in the page cache): per-file overhead and hashing speed (the CPU has SHA extensions) | ~10–20 s | [estimate], measure cold and warm |
| Perceptual hash | full decode + resize ~3–5 ms; less with reduced-size JPEG decoding (`Image.draft()`) | up to ~5–8 min | [estimate] |
| Embeddings, CPU | ~20–50 ms | ~35–85 min | [estimate] |
| Embeddings, one GPU | ~1–2 ms on the GPU (batched); decoding and preprocessing on the CPU at 3–5 ms per image needs at least 3 workers to keep up | ~2–4 min | [estimate] |

**Design consequence:** decoding costs ~20–100× a header read (3–5 ms vs 0.05–0.14 ms), so checks that need full pixels (corruption, "grayscale stored as RGB") should share **one decode pass** rather than each decoding every image again.

The trade-off: one pass couples the checks. Adding a check later, or fixing a bug in one, means rerunning the whole pass. Perceptual hashes only need a tiny image, and Pillow's `Image.draft()` lets the JPEG decoder work at 1/2 to 1/8 scale, which may make a separate hashing pass cheap enough. To measure before deciding.

### Comparisons for duplicate search

| Quantity | Calculation | Result |
|---|---|---|
| All pairs | 101,000² ÷ 2 | ~5.1 billion pairs |
| Train × test pairs only (leakage) | 75,750 × 25,250 | ~1.9 billion pairs |
| Full similarity matrix, float32 | 101,000² × 4 bytes | ~41 GB: can't be held in memory, must be computed **in blocks** |
| Compute for all-pairs cosine similarity | 101,000² × 512 multiply-adds | ~5 × 10¹² multiply-adds: roughly 1–2 minutes on the laptop CPU with NumPy, in blocks [estimate]. This covers the matrix multiply only; selecting the pairs above a threshold among 5 billion scores adds memory traffic and may cost more than the multiply |

### Cost

| Item | Calculation | Result |
|---|---|---|
| GPU rental, embedding run | ~$0.5–1.5 per hour for one on-demand mid-range GPU [estimate, check current provider prices] × at most ~1 h billed (setup + 2–4 min run) | **under ~$2 per run** |
| Moving the data to the GPU machine | 5.12 GB upload at 10–50 MB/s | ~2–9 min, as long as or longer than the embedding run itself. Downloading Food-101 directly on the rented machine avoids it |
| Local passes | laptop | $0 |

### Reading from object storage (for comparison)

| Access pattern | Calculation | Result |
|---|---|---|
| One GET per file, one request at a time (~10 ms latency, ~100 requests/s) | 101,000 ÷ 100 | ~1,010 s ≈ **17 min** |
| One GET per file, 50 requests in parallel | 101,000 ÷ 5,000 requests/s ≈ 20 s, but capped by bandwidth | **~51 s** (bandwidth-bound) |
| Streaming 5.12 GB at ~100 MB/s | 5,120 MB ÷ 100 MB/s | **~51 s** |

The ~20× gap exists only for a client sending one request at a time. Filling the bandwidth with small files takes about *latency × bandwidth ÷ file size* = 1 MB ÷ 50.7 KB ≈ **20 requests in flight**.

---

## 3. Options considered, with their trade-offs

*To be completed as each stage is implemented. Expected decisions:*

- *Decode passes: one shared full-decode pass vs separate passes, with reduced-size decoding for perceptual hashes.*
- *Exact duplicates: MD5 vs SHA-256 vs comparing file sizes first.*
- *Near-duplicates: dHash vs pHash; Hamming distance threshold.*
- *Semantic duplicates: which embedding model; cosine threshold chosen from a precision/recall curve.*
- *Search: exact all-pairs in blocks vs approximate nearest neighbors (FAISS IVF, HNSW).*
- *Storage: CSV vs Parquet vs Lance for the curation table.*
- *Remote reads: per-file objects vs shards; shard size and count.*

## 4. Decision and why

*To fill in, one decision per option above, with the evidence that drove it.*

## 5. Measured result versus estimate

*To fill in as each pass runs: actual time and memory next to the estimates in section 2, and an explanation of every gap larger than 2×.*

| Pass | Estimate | Measured | Gap explained |
|---|---|---|---|
| Header profile | n/a | 14 s cold, ~5 s warm [timed manually] | baseline |
| Full decode | ~5–8 min (1 process) | | |
| SHA-256 | ~10–20 s | | |
| Perceptual hash | up to ~5–8 min | | |
| Embeddings (GPU) | ~2–4 min | | |

---

## 6. What would change at 10 times the scale

### What breaks as the dataset grows

N = number of images. Per-image costs grow with N; pairwise costs grow with N².

| Quantity | Grows with | 1× (Food-101, 101k) | 10× (~1M) | 100× (~10M, Phase 2) |
|---|---|---|---|---|
| Size on disk | N | 5.12 GB | ~51 GB | ~512 GB |
| Whole dataset decoded | N | 71 GB | ~710 GB | ~7.1 TB |
| Embeddings, 512 dims, float32 | N | ~207 MB | ~2.1 GB | ~21 GB (close to the 30 GB of RAM) |
| Full decode, 14 processes | N | ~1 min | ~10 min | ~2 h |
| Embeddings, one GPU | N | ~2–4 min | ~20–40 min | ~3–7 h |
| One GET per file, one at a time | N | ~17 min | ~2.8 h | ~28 h |
| All pairs | N² | 5.1 × 10⁹ | 5.1 × 10¹¹ | 5.1 × 10¹³ |
| Full similarity matrix, float32 | N² | ~41 GB | ~4.1 TB | ~410 TB |
| Exact all-pairs similarity, CPU, in blocks | N² | ~1–2 min | ~2–3 h | ~7–14 days |

Per-image passes grow linearly and stay feasible on one machine with more cores or GPUs. Pairwise comparison grows with the square: at 10×, exact search in blocks still runs in hours; at 100×, it takes weeks of CPU time, which forces approximate nearest-neighbor search (section 3) or a distributed version. Embeddings stop fitting comfortably in RAM around 100× (float16 halves them).

### Storage layout for remote reads

**Where the cost of small files comes from.** Reading Food-101 file by file from object storage, one request at a time, takes about 17 minutes, against about 51 seconds to stream the same 5.12 GB as large files. Every request pays a fixed latency (~10 ms here), while transferring an average 50.7 KB file takes only ~0.5 ms, so most of the time goes to requests, not data. With ~20 or more requests in flight, per-file reads reach the same bandwidth, so at Food-101's size sharding is not needed for speed alone.

**What still argues for shards as the dataset grows:**

- **Request price.** Object stores bill per GET (on the order of $0.0004 per 1,000 GETs for S3 Standard [estimate, check current pricing]): ~$0.04 per pass over Food-101, ~$4 per pass at 10M images, paid again every epoch.
- **Request-rate limits.** S3 documents ~5,500 GETs/s per key prefix [check]; millions of small files need many prefixes or get throttled.
- **Listing.** A LIST call returns at most 1,000 keys: 101 calls for Food-101, 10,000 for 10M images.
- **Client overhead.** Every request costs CPU time and connection handling; for a 50 KB file, that overhead dominates.

**Shard size and count.** Every parallel reader needs its own shard, and several per reader keep shuffling reasonable, so the count follows from the number of readers. Example: 8 GPUs × 4 loader workers = 32 readers; 2 shards each = 64 shards of ~80 MB for Food-101. At 10M images (~512 GB), 64 shards would be 8 GB each, so the shard count grows with the data instead: ~1,000–5,000 shards of 100–500 MB.

**What shards cost.** Fetching one specific image requires an index (which shard, at which position); changing or removing one image means rewriting its whole shard; and shuffling for training is coarser, done by shuffling shard order plus a buffer within shards.

**Rule of thumb.** Sharding pays off when files are smaller than the bandwidth-delay product (latency × bandwidth ≈ 10 ms × 100 MB/s = 1 MB) and the data is read many times from remote storage, for example once per training epoch. Food-101 files (~50 KB) are well below that size, but at 101,000 files parallel requests are enough; for a one-time read from a local disk, sharding is not needed.

*To add: how the distributed version of the pipeline addresses these limits.*
