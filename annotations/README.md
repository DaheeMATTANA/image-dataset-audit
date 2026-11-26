# Annotations

Human judgments about images, keyed by `image_id` (e.g. `apple_pie/1005649`) or by a pair of them, so they can be joined with every per-image table in `outputs/`. Unlike `outputs/`, these files cannot be regenerated: they are committed.

A pipeline stage finds candidates (computed); a person looks at each one and records a verdict here. The verdicts describe the images; they do not remove anything. Which images to drop is a curation rule applied later, on top of these columns.

| File | One row per | Content |
|---|---|---|
| [`flagged_images.csv`](flagged_images.csv) | flagged image | Images with a problem found by eye, from any stage |
| [`exact_duplicate_review.csv`](exact_duplicate_review.csv) | image in an exact-duplicate group | Verdict for every image that shares its pixels with another image |
| [`near_duplicate_review.csv`](near_duplicate_review.csv) | candidate pair | Label for a random sample of near-duplicate candidate pairs, used to measure their precision |

## `flagged_images.csv`

| Column | Values |
|---|---|
| `image_id` | Image key |
| `reason` | `non_food`: not a food photo; `wrong_label`: food photo of another class |
| `found_by` | `<script>:<signal>` that made it a candidate: `profile_images:mode_L` (grayscale), `find_exact_duplicates:pixel` (pixel-duplicate group), `find_near_duplicate_candidates:dhash` / `:phash` (pair drawn for that hash in the near-duplicate review) |
| `labeled_by` | `manual`: checked by eye |
| `note` | What the image shows (`people`, `text`, `place`: a room, kitchen, restaurant or street), or the correct class (`correct: <class>`) |

## `exact_duplicate_review.csv`

Rows are the images of the pixel-duplicate groups in `outputs/exact_duplicates_pixel.csv` (written by `src/find_exact_duplicates.py`), in the same group order.

| Column | Values |
|---|---|
| `image_id` | Image key |
| `pixel_sha256` | Group key: images with the same value have identical pixels |
| `kind` | `within train`, `within test`, or `across train and test` (leakage) |
| `file_sha256` | Same value within a group = byte-identical files; different values = same pixels, different bytes |
| `category` | `reupload`: same food photo, same class; `label_conflict`: same photo, different classes; `non_food`: not a food photo |
| `keep` | `no`: the reviewer judges this image unfit (non-food, or the wrong label of a conflict); empty: no objection. Which copy of a re-upload to drop is left to the curation rule |
| `labeled_by` | `manual`: checked by eye |
| `note` | Reason for `keep=no`: `people`, `text`, `wrong_label` (correct class in `flagged_images.csv`) |

## `near_duplicate_review.csv`

Rows are a random sample of the candidate pairs in `outputs/near_duplicate_candidates_{dhash,phash}.csv` (written by `src/find_near_duplicate_candidates.py`), labeled by eye. `src/measure_near_duplicate_precision.py` reads them to estimate the precision of each hash at each distance.

**Sampling.** For each hash (dHash, pHash) and each distance from 0 to 12 that holds pairs, 10 pairs drawn at random (seed 42), or all of them if fewer. Pixel-identical pairs are left out: they are copies by definition and are covered by `exact_duplicate_review.csv`. A pair drawn for both hashes appears once. This gives 184 pairs: 130 drawn for dHash, 70 for pHash, 16 of them for both.

**Labeling.** The two images were shown side by side, as stored (EXIF orientation not applied, as for the hashes), without class, split or distance, in random order.

| Column | Values |
|---|---|
| `image_id_a`, `image_id_b` | The two images, `image_id_a` < `image_id_b`; on screen, `a` was on the left |
| `drawn_by` | Hash whose sample the pair comes from: `dhash`, `phash`, or `dhash phash`. Each hash's precision uses only its own pairs |
| `dhash_distance`, `phash_distance` | Hamming distance for each hash; empty when the pair is above that hash's radius (12) |
| `kind` | `within train`, `within test`, or `across train and test` (leakage) |
| `label` | See below |
| `labeled_by` | `manual`: checked by eye |
| `note` | Optional words, see below |

| `label` | Meaning | In precision |
|---|---|---|
| `copy` | Same photo, no visible change (re-saved, resized, or a screenshot without visible borders) | positive |
| `near_copy` | Same photo with a visible change (crop, border, brightness, color, text, flip, blur) | positive |
| `same_scene` | Same plate and setting, different shot (e.g. a burst) | negative |
| `same_dish` | Strictly the same kind of food, different photo | negative |
| `unrelated` | Different food, including similar dishes (e.g. steak and pork chop) | negative |
| `unsure` | Cannot tell | left out |

Labels describe what is visible, not how a file was made: a re-encoded or resized copy looks like the original, so both are `copy`.

`note` uses a fixed set of words so they can be counted:

- **Visible change** (with `near_copy`): `crop`, `border`, `brightness`, `color`, `text`, `flip`, `blur`.
- **Not a food photo:** `non_food` followed by what it shows (`people`, `text`, `place`), with `a`, `b` or `both` for which image. These images are also listed in `flagged_images.csv`.
- **Likely reason for a false match** (with a negative label): `similar_border` (both images have a similar frame).
