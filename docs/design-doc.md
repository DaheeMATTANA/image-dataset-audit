# Design doc: Food-101 audit

*Status: draft v0.1. Sections 3–5 cover the stages implemented so far (profile, integrity check, exact duplicates, perceptual hashes, near-duplicate candidates and their precision) and the exact-duplicate hash index; later stages are added as they are implemented.*

*Source labels:*
*- **[measured]**: written by a script in `src/` or `benchmarks/` to `reports/`.*
*- **[experiment]**: written by a script in `experiments/` to `reports/`.*
*- **[per-image table]**: computed from `outputs/profile_images.csv`, not yet in a report.*
*- **[hand-labeled]**: human judgement recorded in `annotations/`.*
*- **[timed manually]**: one timed run, not yet recorded by a script.*
*- **[estimate]**: back-of-envelope, checked against measurements in section 5.*

---

## 1. Problem and constraints

**Problem.** Food-101 is a widely used image classification benchmark. Before a dataset is used for training or evaluation, a curation pipeline needs a precise, file-level picture of its contents. This pipeline builds that picture by auditing the dataset for:

- corrupted or unreadable files
- exact duplicates (identical decoded pixels; byte-identical files are a subset)
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
- **Milestone:** audit report and this design doc v1 at the end of the audit.
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
| Embeddings (512 dims, float32) | 101,000 × 512 × 4 bytes | ~207 MB (~103 MB in float16): fits in RAM. 512 dims assumes a CLIP ViT-B encoder (e.g. CLIP ViT-B/32); CLIP ViT-L/14 outputs 768 dims (~310 MB) |
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

| Decision | Options | Trade-off |
|---|---|---|
| Definition of an exact duplicate | Same file bytes (file hash) vs same decoded pixels (pixel hash) | The file hash is cheap (no decode) but misses copies that differ only in metadata; the pixel hash needs a full decode but matches what a training loader sees |
| Decode passes | One shared full-decode pass vs one pass per check | One pass decodes each image once; it couples the checks, so a fix to one check reruns the whole pass |
| Parallelism | Threads vs processes | Threads share memory and start fast, but Python code between C calls holds the GIL; processes scale on CPU-bound work but each one starts and holds its own memory |
| Hash width in the index | 32, 64 or 256 bits of SHA-256 | Fewer bits save memory per entry but let different images share a value by chance |
| Where the hash index lives | Dict in memory on one machine; sorted file + merge; key-value store; sharding by hash prefix | Compared in section 6, at 10M and 1B images |
| Near-duplicate search | Compare every pair of perceptual hashes (exact) vs an index or approximate search that skips most pairs | Every pair is exact and simple but grows with N²; an index scales further but can miss pairs, and its recall can only be measured against an exact run |
| Pair comparison code | Python loop over pairs vs NumPy, one row against all later rows | The loop needs no array setup but runs Python code for every pair; NumPy makes one call per row and does the pairs in compiled code |
| Candidate radius | One Hamming distance radius for all hashes vs one per hash | One radius is simpler, but the three hashes spread pairs very differently at small distances (section 4) |
| Pairs to label | Random pairs from all candidates vs a fixed number per distance | A random sample follows the counts: 84% of dHash candidates lie at distance 11–12 and 0.2% at distance 0–5, where the threshold is decided [measured], so 200 random pairs would hold almost none of the copies. A fixed number per distance covers every distance, but each distance must then be weighted by its pair count |
| Near-duplicate rule | dHash or pHash; one hash or both; distance threshold | A lower threshold returns fewer false pairs but misses more copies; a second hash finds a few more copies at the cost of more false pairs |

*Still to decide, as the later stages are implemented:*

- *Semantic duplicates: which embedding model; cosine threshold chosen from a precision/recall curve.*
- *Search: exact all-pairs in blocks vs approximate nearest neighbors (FAISS IVF, HNSW).*
- *Storage: CSV vs Parquet vs Lance for the curation table.*
- *Remote reads: per-file objects vs shards; shard size and count.*

## 4. Decision and why

| Decision | Choice | Evidence |
|---|---|---|
| Definition of an exact duplicate | Same pixel hash (SHA-256 of the RGB pixels after `convert("RGB")`, no EXIF rotation applied); byte-identical files are reported as a subset | 58 pixel-hash groups vs 8 file-hash groups [measured]; in all 52 compared pairs of pixel-identical files with different bytes, only metadata differs (dates, GPS, Photoshop/XMP blocks) [experiment] |
| Hash function | SHA-256 | MD5 allows deliberately built collisions, which matters once data comes from the open web; on this CPU (SHA extensions), SHA-256 is also faster: ~44 vs ~97 µs per 50.7 KB [timed manually] |
| Decode passes | One full-decode pass computes the corruption check, the file hash and the pixel hash | ~1.46 ms per image for all three together, one process (687 img/s on 5,000 images) [measured] |
| Perceptual hashes | Separate pass with its own decode, not merged into the integrity pass | Hashing is a different concern from integrity, and the hash stage is rerun while hashes and thresholds are tuned; the second decode costs ~20 s at 101,000 images [timed manually]. At 10M images (~37 min per extra decode pass), one decode shared by all CPU signals is the better choice (section 6) |
| Near-duplicate search at Food-101's size | Every pair, NumPy one row at a time, one process, for each of the three hashes | 5,100,449,500 pairs per hash in 8.1–8.5 s, 11.0–11.4 s with candidate pairs and same-class counts [timed manually]; NumPy is 24.5× faster than the Python loop on 5,000 hashes, with the same counts [experiment]. An exact run is also the reference for measuring the recall of faster methods later; workers would save only seconds here |
| Candidate radius | dHash 12, pHash 12; aHash counted, no candidates | Pairs per distance stay in the tens up to distance 4 (dHash) and 6 (pHash), then grow; aHash already has 66,300 pairs within distance 4 [measured]. Radius 12 lies past both ranges, so hand labels can test pairs on both sides of a threshold: 125,465 dHash and 36,886 pHash candidate pairs [measured] |
| Labeled sample | 10 pairs per hash and distance 0–12, 184 pairs in all, labeled without class, split or distance on screen; the 62 pixel-identical pairs count as known copies; each distance is weighted by its pair count | Small distances hold 15–108 pairs each, so 10 labels cover a large share of them; at large distances 10 labels only bound the share of copies, and the same-class count bounds it further [hand-labeled, measured] |
| Near-duplicate rule | pHash distance ≤ 4; dHash kept as a reported signal | pHash ≤ 4 returns 145 pairs, and all 92 checked ones are copies (62 pixel-identical, 30 labeled) [measured]. dHash ≤ 4 returns 171 pairs at an estimated 86.4% [measured]. One step further adds 30 pairs at 20% (pHash 6) or 63 pairs at 20% (dHash 5) [measured]. The threshold is chosen on the same labels, so pHash ≤ 4's precision is an upper reading |
| Same-class count per distance | Counted with the pair counts, in the same pass | Share of same-class pairs: 98% at distance 0 (dHash, pHash), below 10% from distance 5 (dHash) and 8 (pHash), 0.99% over all pairs [measured]: shows without labels where copies give way to unrelated pairs; costs ~+1.8 s per hash [timed manually] |
| Parallelism | Processes, 14 workers for the full run | The check is CPU-bound: CPU time = wall time on 2,000 images [timed manually]. Threads match processes up to 4 workers, peak at 6 (2,731 img/s) and fall after; processes reach 3,713 img/s at 14 [measured] |
| Hash width | Full 256-bit hashes | At 32 bits, different images already share a value at Food-101's size: 3 (file hash) and 4 (pixel hash) chance pairs [experiment]; see section 6 for 10M and 1B |
| Hash index at Food-101's size | In-memory group-by on the per-image hash table (`outputs/integrity.csv`) | 100,940 unique pixel hashes; as a Python dict, ~18 MB (176 bytes per entry) [experiment]: far below the 30 GB of RAM |

## 5. Measured result versus estimate

Actual time and memory next to the estimates in section 2; every gap larger than 2× is explained.

| Pass | Estimate | Measured | Gap explained |
|---|---|---|---|
| Header profile | n/a | 14 s cold, ~5 s warm [timed manually] | baseline |
| Full decode + file and pixel SHA-256 | ~3–5 ms per image; ~5–8 min (1 process) | ~1.46 ms per image (687 img/s, 1 process, 5,000 images) [measured], ~2.5 min for 101,000 images by extrapolation; 22.4 s with 14 processes, cold page cache [timed manually]; 21.4 s (14) and 19.0 s (20 processes), warm page cache [timed manually] | 2–3× below the estimate, which was a generic decode cost, not derived from a measurement on these 512 px images |
| SHA-256 | ~10–20 s (separate pass) | not timed separately: runs inside the decode pass | n/a |
| Hash index in memory | ~165 bytes per entry (mock estimate) | 176 bytes per entry (32-byte digest, image ID as value) [experiment] | within 7% |
| Perceptual hash (aHash, dHash, pHash; own full decode) | up to ~5–8 min | 38.6 s with 14 processes, 33.3 s with 20, warm page cache [timed manually]; 1.8× the integrity check under the same conditions | The estimate is for 1 process; the 1-process cost per image is not measured yet |
| All-pairs Hamming distances, one hash, one process | ~5.2 min with a Python loop, ~13 s with NumPy, extrapolated from 5,000 hashes (16.5M and 403M pairs/s) [experiment] | 8.1–8.5 s with NumPy, counts only [timed manually]; the loop was not run at 101,000 | NumPy is 1.6× faster than extrapolated: each row pays a fixed cost for its Python calls, shared by more pairs when rows are longer (50,500 hashes per row on average at 101,000 images, 2,500 at 5,000) |
| Near-duplicate precision within distance 12 | At most the share of same-class pairs, since copies of one photo almost always share a class: 2,875 of 125,465 dHash pairs (2.3%), 885 of 36,886 pHash pairs (2.4%) [measured] | 0.2% dHash, 0.4% pHash [measured] | About 10× below the bound: at large distances, most same-class pairs are different photos of the same dish, not copies |
| Embeddings (GPU) | ~2–4 min | | |

---

## 6. What would change at 10 times the scale

### What breaks as the dataset grows

N = number of images. Per-image costs grow with N; pairwise costs grow with N².

| Quantity | Grows with | 1× (Food-101, 101k) | 10× (~1M) | 100× (~10M) |
|---|---|---|---|---|
| Size on disk | N | 5.12 GB | ~51 GB | ~512 GB |
| Whole dataset decoded | N | 71 GB | ~710 GB | ~7.1 TB |
| Embeddings, 512 dims, float32 | N | ~207 MB | ~2.1 GB | ~21 GB (close to the 30 GB of RAM) |
| Full decode + both hashes, 14 processes | N | 22 s [timed manually] | ~4 min | ~37 min |
| Perceptual hashes (separate decode pass), 14 processes | N | 39 s [timed manually] | ~6.5 min | ~64 min |
| Hash index, Python dict (32-byte digest + image ID) | N | ~18 MB | ~176 MB | ~1.8 GB [experiment, projected] |
| Expected chance collisions, 32-bit hash | N² | 1.19 (3–4 observed) | ~116 | ~11,642 [experiment] |
| Embeddings, one GPU | N | ~2–4 min | ~20–40 min | ~3–7 h |
| One GET per file, one at a time | N | ~17 min | ~2.8 h | ~28 h |
| All pairs | N² | 5.1 × 10⁹ | 5.1 × 10¹¹ | 5.1 × 10¹³ |
| All-pairs Hamming distances, one hash, one process | N² | ~8 s [timed manually] | ~14 min | ~22 h |
| dHash candidates within distance 12, if the same share of pairs | N² | 125,465 pairs [measured], ~5 MB | ~12.5M pairs, ~0.5 GB | ~1.25B pairs, ~53 GB [estimate, ~42 bytes per pair in CSV] |
| Full similarity matrix, float32 | N² | ~41 GB | ~4.1 TB | ~410 TB |
| Exact all-pairs similarity, CPU, in blocks | N² | ~1–2 min | ~2–3 h | ~7–14 days |

Per-image passes grow linearly and stay feasible on one machine with more cores or GPUs. Pairwise comparison grows with the square: at 10×, exact search in blocks still runs in hours; at 100×, it takes weeks of CPU time, which forces approximate nearest-neighbor search (section 3) or a distributed version. Embeddings stop fitting comfortably in RAM around 100× (float16 halves them).

For near-duplicates, comparing every pair of 64-bit hashes stays practical up to about 10× on one machine. At 100×, it takes about a day per hash in one process (workers divide this by at most the number of cores), and the candidate list, ~53 GB if the same share of pairs falls within the radius, no longer fits in memory: it has to be written in chunks. Past that scale, an index that avoids comparing every pair is needed, with its recall measured against the exact run at Food-101's size.

A distance threshold does not keep its precision on a larger dataset. If the share of images that have a copy stays the same, the number of copy pairs grows with N, while pairs of unrelated images that land close by chance grow with N². At dHash ≤ 4, Food-101 has an estimated 147.7 copies and 23.3 other pairs (171 pairs at 86.4%) [measured]. At 10×, that gives ~1,480 copies and ~2,330 other pairs, a precision of ~39%; at 100×, ~14,800 copies and ~233,000 other pairs, ~6% [estimate]. pHash ≤ 4 had no false pair among the 92 checked, so its growth can't be projected from this sample; a single false pair per 145 at Food-101's size would already become ~10,000 at 100×, against ~14,500 copies. The threshold has to be measured again on labels drawn from the larger dataset, and is likely to tighten.

### Where the exact-duplicate hash index lives

**The index.** Exact deduplication keeps an index from pixel hash to image: a hash already in the index is a duplicate. Two steps are involved, with very different costs:

- **Computing the hashes** needs a full decode of every image (~37 min at 10M with 14 processes, section above). Stage 2 saves the result per image (`outputs/integrity.csv`), so this work is never repeated.
- **Building the index** only reads those saved hashes: 10M rows, seconds to a minute [estimate].

So the index itself is cheap to rebuild. What a crash must not lose is the hashing work, which needs per-shard progress (a restart skips the shards already done) whatever the index option.

**Size.** Raw data per entry: 32-byte hash + image ID (~25 bytes [estimate]) ≈ 57 bytes, so ~570 MB at 10M and ~57 GB at 1B as a file. In memory as a Python dict, an entry takes 176 bytes [experiment]: most of it is Python object overhead (each object carries a fixed header of about 33–41 bytes).

**Options at 10M and 1B images:**

| | 1. Dict, one machine | 2. Sorted file + merge | 3. Key-value store | 4. Shard by hash prefix |
|---|---|---|---|---|
| Memory at 10M | ~1.8 GB [experiment, projected] | small: works in chunks; ~570 MB on disk | ~1–1.5 GB in the store's RAM, or on SSD [estimate] | 256 shards of ~39,000 entries: ~7 MB each as a dict |
| At 1B | ~176 GB: too big for one machine | ~57 GB on disk: fine | needs a cluster | ~690 MB per shard: fine |
| Speed | fastest: one memory lookup per image | one sort, O(N log N): seconds to minutes at 10M | one network round trip per lookup (batched) | shards run in parallel |
| New data each week | rebuilt from the saved hashes each run, or kept alive between runs | the new week's batch is sorted and merged into the old file in one linear pass | built for it: each new item is answered at once | only the touched shards change |
| Many writers | no: one process | no conflicts: each worker writes its own file | yes: "insert if absent" is atomic, the store decides which writer wins | no conflicts: one owner per shard |
| After a crash | lost; rebuilt from the saved hashes in seconds to a minute | nothing lost if each file is written to a temporary name, then renamed (an atomic operation on Linux: readers see the old file or the new one, never a partial one) | persistent, if configured | depends on the option used inside each shard |
| What to operate | nothing | nothing: files only | a service to run, monitor and pay for all the time | a shuffle step that sends each row to its shard |
| Cost | none extra | storage only | an always-on service | compute only during runs |

**Why sharding by hash prefix is correct for exact duplicates.** Copies of the same image have the same hash, so they always land in the same shard, and each shard can be deduplicated on its own. SHA-256 prefixes are uniform, so shards have the same size (no skew). This does not carry over to near-duplicates or semantic duplicates, where similar images have different hash values or embeddings.

**Full hashes, not truncated ones.** Expected chance collisions among n hashes of b bits: n(n − 1) / 2 ÷ 2^b. Measured at 101,000 images, the formula matches within one standard deviation at 20–28 bits [experiment]. At 32 bits: 3–4 chance pairs at 101,000 images, ~11,642 expected at 10M. At 64 bits: about 1 in 369,000 at 10M, but 1 in 37 at 1B. Truncation also saves little memory: a 4-byte key instead of 32 bytes saves 28 of 176 bytes per entry (16%) [experiment]. Every format measured stays between 145 and 203 bytes per entry, so between 145 and 203 GB at 1B.

**Decision, by condition.** The pipeline's future needs (weekly updates, number of writers) are not fixed yet, so the choice is stated per condition:

| Condition | Choice | Why |
|---|---|---|
| One batch run, at 10M | 1. Dict in memory, built from the saved hashes | ~1.8 GB fits easily; nothing to operate; rebuilt in seconds after a crash |
| New data every week | 2. Sorted file + merge | The index persists between runs as a file; each new batch is merged in one linear pass; no service to run |
| Many workers writing at once | 4. Shard by hash prefix, a dict or sorted file per shard | One owner per shard, no write conflicts; ~7 MB per shard at 10M |
| Single lookups at any moment, with low latency | 3. Key-value store | The only option that answers one new item at once, at the cost of an always-on service |
| 1B images, any condition | 2 or 4 | No in-memory format fits on one machine (145–203 GB) |

**What changes at 1B.** Shrinking entries cannot fix the 1B problem: the gain is at most ~18% (row number instead of image ID, 176 → 145 bytes), and truncation causes false duplicates. The fix is changing the option: move the index to disk (2) or split it by hash prefix (4).

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
