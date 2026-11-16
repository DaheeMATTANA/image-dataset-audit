# %% [markdown]
# # dHash by hand: which edits keep an image close to the original?
# One Food-101 image, four edited versions of it and two other images.
# Each is reduced to a 64-bit dHash; the Hamming distance (bits that differ)
# to the original says how close dHash considers them.
#
# Writes reports/experiment_dhash.csv.

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
# `distance`: the image as it is. `best_of_4_rotations`: the smallest distance
# over the image at 0°, 90°, 180° and 270°.

# %%
base = dhash(original)
rows = []
for name, img in cases.items():
    rows.append(
        {
            "case": name,
            "distance": hamming(base, dhash(img)),
            "best_of_4_rotations": min(
                hamming(base, dhash(turned)) for turned in rotations(img)
            ),
        }
    )
table = pd.DataFrame(rows)
print(table.to_string(index=False))

out = REPORTS / "experiment_dhash.csv"
table.to_csv(out, index=False)
print(f"Saved: {out}")

# %% [markdown]
# We discovered that the other class image approaches when taken the best of
# 4 rotations (chance effect).
# Trade-off: we catch rotated copies (33 -> 0) but false matches are possible
# with unrelated images.

# %%
