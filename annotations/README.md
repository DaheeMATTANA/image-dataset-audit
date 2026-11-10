# Annotations

Human judgements about images, keyed by `image_id` (e.g. `apple_pie/1005649`), so they can be joined with every per-image table in `outputs/`. Unlike `outputs/`, these files cannot be regenerated: they are committed.

A pipeline stage finds candidates (computed); a person looks at each one and records a verdict here. The verdicts describe the images; they do not remove anything. Which images to drop is a curation rule applied later, on top of these columns.

| File | One row per | Content |
|---|---|---|
| [`flagged_images.csv`](flagged_images.csv) | flagged image | Images with a problem found by eye, from any stage |
| [`exact_duplicate_review.csv`](exact_duplicate_review.csv) | image in an exact-duplicate group | Verdict for every image that shares its pixels with another image |

## `flagged_images.csv`

| Column | Values |
|---|---|
| `image_id` | Image key |
| `reason` | `non_food`: not a food photo; `wrong_label`: food photo of another class |
| `found_by` | `<script>:<signal>` that made it a candidate: `profile_images:mode_L` (grayscale), `find_exact_duplicates:pixel` (pixel-duplicate group) |
| `labeled_by` | `manual`: checked by eye |
| `note` | What the image shows (`people`, `text`), or the correct class (`correct: <class>`) |

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
