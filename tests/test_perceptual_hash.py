"""Tests for the perceptual hash helpers, on synthetic images (no data/ needed)."""

import imagehash
import numpy as np
import pytest
from PIL import Image

from image_dataset_audit.perceptual_hash import dhash, hamming

# Test bound for a resized copy; not the pipeline's near-duplicate threshold.
RESIZE_BOUND = 10

WIDTH, HEIGHT = 512, 384  # same long side as Food-101 images


@pytest.fixture
def noise() -> Image.Image:
    """Random pixels, fixed seed: the hard case for dHash.

    Neighboring cells average to almost the same brightness, so small
    changes can flip bits (a resize flips 4).
    """
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


@pytest.fixture
def gradient() -> Image.Image:
    """Brightness rising from left to right: every bit is 1.

    Gives the largest possible hash, the edge case for the 64-bit check.
    """
    row = np.linspace(0, 255, WIDTH).astype(np.uint8)
    gray = np.tile(row, (HEIGHT, 1))  # shape (HEIGHT, WIDTH)
    return Image.fromarray(gray).convert("RGB")


def test_identical_image_has_distance_zero(noise):
    assert hamming(dhash(noise), dhash(noise.copy())) == 0


def test_resized_copy_is_close(noise):
    small = noise.resize((WIDTH // 2, HEIGHT // 2), Image.Resampling.LANCZOS)
    assert hamming(dhash(noise), dhash(small)) <= RESIZE_BOUND


def test_matches_imagehash(noise):
    assert dhash(noise) == int(str(imagehash.dhash(noise)), 16)


def test_hash_fits_in_64_bits(gradient):
    assert dhash(gradient) == 2**64 - 1  # all 64 bits set, nothing beyond


def test_hamming_is_symmetric():
    a, b = 0b1011_0000, 0b0110_1111
    assert hamming(a, b) == hamming(b, a)
