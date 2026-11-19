"""Tests for the perceptual hash stage, on synthetic JPEGs written to tmp_path."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from compute_perceptual_hashes import compute_hashes

WIDTH, HEIGHT = 512, 384


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    """Where a test writes its image: <class>/<id>.jpg, so image_id is "pie/1"."""
    path = tmp_path / "pie" / "1.jpg"
    path.parent.mkdir()
    return path


@pytest.fixture
def reverse_gradient() -> Image.Image:
    """Brightness falling from left to right: every bit is 0.

    Checks for it to have 16 hex.
    """
    row = np.linspace(0, 255, WIDTH).astype(np.uint8)
    gray = np.tile(row, (HEIGHT, 1))  # shape (HEIGHT, WIDTH)
    img = Image.fromarray(gray).convert("RGB")
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


@pytest.fixture
def noise() -> Image.Image:
    """Random RGB pixels, fixed seed: no structure the hashes could share."""
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def test_hashes_are_16_hex_chars(image_path, reverse_gradient):
    """An all-zero dHash keeps its leading zeros: "0" * 16, not "0"."""
    reverse_gradient.save(image_path)
    row = compute_hashes(image_path)
    assert row["dhash"] == "0" * 16
    for col in ("ahash", "dhash", "phash"):
        assert len(row[col]) == 16


def test_broken_file_gives_empty_hashes(image_path):
    """An undecodable file doesn't raise: hashes are None, image_id is kept."""
    image_path.write_text("not a jpeg")
    row = compute_hashes(image_path)
    assert row["image_id"] == "pie/1"
    assert row["ahash"] is None
    assert row["dhash"] is None
    assert row["phash"] is None


def test_truncated_file_gives_empty_hashes(image_path, noise):
    """A cut-off JPEG fails instead of being padded with gray."""
    noise.save(image_path)
    data = image_path.read_bytes()
    image_path.write_bytes(data[: len(data) // 2])  # keep the first half only
    row = compute_hashes(image_path)
    assert row["ahash"] is None
    assert row["dhash"] is None
    assert row["phash"] is None
