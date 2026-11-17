"""
Perceptual hashes and the distance between them.
"""

import numpy as np
from PIL import Image


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
