# image-dataset-audit

**How clean is a widely used image dataset, really, and does it matter for the models trained on it?**

This repository audits [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), a public benchmark of 101,000 food photos across 101 classes. It is used everywhere, from training tutorials to the evaluation suites of large vision-language models. The audit looks for the problems that quietly distort results: broken files, exact duplicates, near-duplicates, and test images that also appear in the training set. It then measures how much those problems change a model's reported accuracy.

The goal is not only a list of bad files, but a measured answer: *if we remove the leakage, how much does accuracy drop?*

> **Status:** in progress (started October 2026). Results are added to [Findings](#findings) as each step is completed.

---

## Why this matters

Benchmark numbers are only as trustworthy as the data behind them.

- **Duplicates waste compute and bias training.** The same photo seen many times counts more than it should.
- **Train/test leakage inflates accuracy.** If a test image (or a resized copy of it) is also in the training set, the model is partly graded on memory, not generalization.
- **Label noise is a known property of Food-101.** The authors left the training set uncleaned on purpose. How much noise there actually is has to be measured.

Data curation (finding and removing bad data, then proving it helps) is now a central part of how modern vision and multimodal models are built. This project applies that loop at a small, fully reproducible scale on a laptop.

---

## Questions this project answers

| # | Question | Method | Output |
|---|---|---|---|
| 1 | What does the dataset really contain? | Profile counts, splits, formats, color modes, resolutions, file sizes | Data profile |
| 2 | Which files are broken or exact duplicates? | Full decode check with Pillow, SHA-256 hashing | Corruption and exact-duplicate report |
| 3 | Which images are the same photo resized or re-encoded? | Perceptual hashing (dHash, pHash), Hamming distance, 200 hand-labelled pairs | Near-duplicate list with measured precision |
| 4 | How does a model "see" an image? | Embeddings from a pretrained vision encoder (open_clip) | Embeddings for the full dataset |
| 5 | When are two images semantically "close"? | Cosine similarity, threshold chosen from a precision/recall curve | Semantic duplicates with precision and recall |
| 6 | How do we compare 101k × 101k images fast enough? | Exact vs approximate nearest-neighbour search (FAISS) | Speed/recall trade-off table |
| 7 | How are the results stored and reused? | Columnar storage: one row per image, one column per curation signal | Curation table (Parquet) |
| 8 | **Does leakage inflate accuracy?** | Linear probe on embeddings, original vs deduplicated split, 3 seeds | Ablation result with variance |
| 9 | How would this be explained to an architect? | Design doc: constraints, options, decision, measured trade-offs | Audit report and design doc |

---

## Hypotheses (written before looking at the data)

Each claim from the [original paper](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_29) is treated as something to verify, not something to trust. My guesses are recorded here so they can be compared honestly with the measurements.

| Dimension | Paper says | My first guess | Measured |
|---|---|---|---|
| Classes | 101, balanced | 101, exactly 1,000 images each | _pending_ |
| Splits | 750 train / 250 test per class | 75,750 / 25,250, no image in both lists | _pending_ |
| Format | not stated | 100% JPEG, nearly all RGB, a few grayscale | _pending_ |
| Resolution | long side ≤ 512 px, smaller images excluded | 100% have a long side of exactly 512 px | _pending_ |
| Exact duplicates | not stated | < 0.5% | _pending_ |
| Near-duplicates | not stated | 1–3%, some crossing train/test | _pending_ |
| Label noise (train) | "some amount of noise", left on purpose | 3–5% wrong or non-food images | _pending_ |
| Test set | manually cleaned | much cleaner than train | _pending_ |

---

## Findings

_Results will appear here as each question is answered, with links to the full reports in [`reports/`](reports/)._

---

## Repository structure

```
image-dataset-audit/
├── src/            # audit scripts: profiling, hashing, embeddings, dedup, ablation
├── tests/          # pytest tests for the hashing and dedup logic
├── reports/        # generated tables, plots and the written audit report
├── data/           # Food-101 lives here locally (git-ignored, ~5 GB)
├── pyproject.toml  # dependencies
└── uv.lock         # exact pinned versions, for reproducibility
```

---

## Reproducing the audit

Requirements: Linux or macOS, [uv](https://docs.astral.sh/uv/), about 10 GB of free disk space.

```bash
# 1. Clone and install the exact environment
git clone https://github.com/<your-username>/image-dataset-audit.git
cd image-dataset-audit
uv sync

# 2. Download Food-101 (~5 GB) into data/
mkdir -p data
wget -P data http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf data/food-101.tar.gz -C data

# 3. Run the audit scripts (added step by step)
uv run python src/<script>.py
```

Every script reads from `data/` and writes its results to `reports/`, so all numbers in this README can be regenerated.

---

## Tools

Python 3.12 · uv · Pillow · pandas · NumPy · matplotlib · imagehash · pytest · PyTorch and open_clip · FAISS · pyarrow / Parquet

---

## What comes next

This audit is the first stage of a larger project on data curation for multimodal models. The same pipeline will later be scaled to around 10 million images on multiple machines, with throughput and cost measured, and then extended to video.

---

## Dataset and citation

Food-101 was created by Lukas Bossard, Matthieu Guillaumin and Luc Van Gool (ETH Zürich). The images come from foodspotting.com. This repository does not redistribute any images; please check the [official dataset page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) for terms of use.

```bibtex
@inproceedings{bossard14,
  title     = {Food-101 -- Mining Discriminative Components with Random Forests},
  author    = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year      = {2014}
}
```
