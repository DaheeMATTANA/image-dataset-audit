# %% [markdown]
# # Perceptual hashes on edited copies: aHash, dHash and pHash
# One Food-101 image, four edited versions of it and two other images.
# Each is reduced to a 64-bit hash (own dHash, imagehash's aHash and pHash);
# the Hamming distance (bits that differ) to the original says how close
# each hash considers them.
#
# Writes reports/experiment_perceptual_hash.csv.

# %%
import io

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from image_dataset_audit.dataset import IMAGES, REPORTS

# %% [markdown]
# ## The six cases

# %%
original = Image.open(IMAGES / "apple_pie" / "1005649.jpg").convert("RGB")
w, h = original.size  # long side is 512 px


def reencode(img: Image.Image, quality: int) -> Image.Image:
    """Save as JPEG in memory at `quality` and decode it again."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def load(image_id: str) -> Image.Image:
    return Image.open(IMAGES / f"{image_id}.jpg").convert("RGB")


cases = {
    "resized to 256 px": original.resize((w // 2, h // 2), Image.Resampling.LANCZOS),
    "re-encoded at q50": reencode(original, 50),
    "cropped 10% per side": original.crop((w // 10, h // 10, w - w // 10, h - h // 10)),
    "rotated 90°": original.transpose(Image.Transpose.ROTATE_90),
    "same class, other image": load("apple_pie/1011328"),
    "other class": load("pizza/1001116"),
}

# %% [markdown]
# ## dHash, Hamming distance and rotations


# %%
def dhash(img: Image.Image) -> int:
    """64-bit difference hash: is each pixel brighter than its left neighbor?"""
    small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small)  # shape (8, 9): 8 rows, 9 columns
    bits = pixels[:, 1:] > pixels[:, :-1]  # shape (8, 8): right > left
    value = 0
    for bit in bits.flatten():  # row by row, first bit is the highest
        value = value * 2 + int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Number of bit positions where a and b differ."""
    return (a ^ b).bit_count()


def rotations(img: Image.Image) -> list[Image.Image]:
    """The image at 0°, 90°, 180° and 270°."""
    turns = [
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
    ]
    return [img] + [img.transpose(turn) for turn in turns]


# %% [markdown]
# ## aHash and pHash from imagehash
# Converted to int, like dHash, so the same `hamming` compares all three.


# %%
def ahash(img: Image.Image) -> int:
    """64-bit average hash from imagehash, as an int."""
    return int(str(imagehash.average_hash(img)), 16)


def phash(img: Image.Image) -> int:
    """64-bit DCT-based perceptual hash from imagehash, as an int."""
    return int(str(imagehash.phash(img)), 16)


HASHES = {"dhash": dhash, "ahash": ahash, "phash": phash}

# %% [markdown]
# ## Check against the imagehash library
# Same 64 bits as imagehash.dhash on every image, in all 4 rotations.

# %%
for name, img in {"original": original, **cases}.items():
    matches = [
        dhash(turned) == int(str(imagehash.dhash(turned)), 16)  # hex string -> int
        for turned in rotations(img)
    ]
    print(f"{name:25} same as imagehash (4 rotations): {all(matches)}")

# %% [markdown]
# ## Distances to the original
# `dhash`, `ahash`, `phash`: distance to the original as is.
# `<hash>_best_of_4`: the smallest distance over the image at 0°, 90°, 180° and 270°.

# %%
base = {name: fn(original) for name, fn in HASHES.items()}

rows = []
for case, img in cases.items():
    turned = rotations(img)  # index 0 is the image as it is
    row = {"case": case}
    for name, fn in HASHES.items():
        distances = [hamming(base[name], fn(t)) for t in turned]
        row[name] = distances[0]
        row[f"{name}_best_of_4"] = min(distances)
    rows.append(row)

table = pd.DataFrame(rows)
print(table.to_string(index=False))

out = REPORTS / "experiment_perceptual_hash.csv"
table.to_csv(out, index=False)
print(f"Saved: {out}")

# %% [markdown]
# We discovered that the other class image approaches when taken the best of
# 4 rotations (chance effect).
# Trade-off: we catch rotated copies (33 -> 0) but false matches are possible
# with unrelated images.

# %% [markdown]
# ## Gap per hash

# %%
# Rotated copy left out: no hash survives a 90° rotation (28-33 bits, near the
# ~32 of unrelated hashes); rotation is handled by the best-of-4 columns.
different_image = ["same class, other image", "other class"]
same_image = [c for c in cases if c not in different_image and c != "rotated 90°"]

by_case = table.set_index("case")
# gap = (smallest "different image" distance) - (largest "same image" distance)
gap = by_case.loc[different_image].min() - by_case.loc[same_image].max()
print(gap.to_string())

# %% [markdown]
# Gap (smallest different-image distance - largest same-image distance, rotated
# copy left out): 13 / 18 / 20 bits for dHash / aHash / pHash, set by the crop
# and the same-class image. Best of 4 rotations brings the rotated copy to 0
# with the same gaps. One image only: the gaps describe this image, not the
# hashes in general.
