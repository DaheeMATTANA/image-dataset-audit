import pytest

from image_dataset_audit import dataset
from image_dataset_audit.dataset import list_images, sample_images


@pytest.fixture
def fake_images(tmp_path, monkeypatch):
    """50 empty files in 5 class folders, with IMAGES pointed at them."""
    for c in range(5):
        folder = tmp_path / f"class_{c}"
        folder.mkdir()
        for i in range(10):
            (folder / f"{i}.jpg").touch()
    monkeypatch.setattr(dataset, "IMAGES", tmp_path)
    return tmp_path


def test_same_seed_gives_same_sample(fake_images):
    assert sample_images(10, seed=1) == sample_images(10, seed=1)


def test_different_seed_gives_different_sample(fake_images):
    assert sample_images(10, seed=1) != sample_images(10, seed=2)


def test_sample_larger_than_dataset_returns_every_file(fake_images):
    sample = sample_images(1000)
    assert len(sample) == 50
    assert sorted(sample) == list_images()


def test_empty_folder_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "IMAGES", tmp_path)
    with pytest.raises(FileNotFoundError):
        list_images()


@pytest.mark.skipif(not dataset.IMAGES.is_dir(), reason="Food-101 not in data/")
def test_default_sample_is_unchanged():
    # Pinned from the sampling code that produced the committed reports. Fails if
    # the listing order, Python's random.sample, or the files on disk change.
    first = [f"{p.parent.name}/{p.stem}" for p in sample_images(5000)[:3]]
    assert first == [
        "red_velvet_cake/372523",
        "carrot_cake/31925",
        "beef_carpaccio/1864287",
    ]
