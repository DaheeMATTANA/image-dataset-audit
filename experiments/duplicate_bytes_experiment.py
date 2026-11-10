"""Why do pixel-identical copies differ in bytes?

Takes the pixel-duplicate groups whose files are not byte-identical (more than one
file_sha256 in annotations/exact_duplicate_review.csv) and compares the copies:

1. What a JPEG file holds besides the pixels (size, EXIF, orientation, ICC profile,
   progressive coding, comment, quantization tables, subsampling, metadata blocks):
   in how many groups does each one differ?
2. For copies of the same size: how many bytes differ, where, and are they all
   inside the APP1 (EXIF) segment? Compares each copy with its group's first image.
3. For every compared pair: which EXIF tags differ, and are the blocks the decoder
   needs (tables, frame header, image data) identical?

Prints the tables.

    uv run python experiments/duplicate_bytes_experiment.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ExifTags, Image, JpegImagePlugin

from check_integrity import IMAGES
from dataset_utils import ROOT

REVIEW = ROOT / "annotations" / "exact_duplicate_review.csv"
# Tags that hold the position of another EXIF block, not a value
POINTER_TAGS = {ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo, ExifTags.IFD.Interop}


def jpeg_facts(image_id: str) -> dict:
    """What a JPEG file holds besides the pixels: size, metadata, coding."""
    path = IMAGES / f"{image_id}.jpg"
    with Image.open(path) as im:
        return {
            "image_id": image_id,
            "size_bytes": path.stat().st_size,
            "exif_bytes": len(im.info.get("exif", b"")),
            "orientation": im.getexif().get(274),
            "icc": "icc_profile" in im.info,
            "progressive": bool(im.info.get("progressive", 0)),
            "comment": im.info.get("comment"),
            "quant": im.quantization,
            "subsampling": JpegImagePlugin.get_sampling(im),
            "info_keys": tuple(sorted(im.info)),
        }


def app1_span(data: bytes) -> tuple[int, int] | None:
    """Byte range of the first APP1 (EXIF) segment, or None."""
    start = data.find(b"\xff\xe1")
    if start == -1:
        return None
    length = int.from_bytes(data[start + 2 : start + 4], "big")  # includes these 2
    return start, start + 2 + length


def image_part(data: bytes) -> bytes:
    """The file without its metadata blocks: what the decoder needs.

    Walks the JPEG blocks from the start. APP0-APP15 (0xE0-0xEF) and COM (0xFE)
    hold metadata (JFIF, EXIF, ICC, Photoshop, XMP, comments) and are dropped;
    tables, frame header and image data are kept.
    """
    kept, i = [], 2  # skip the start-of-image marker (FFD8)
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: image data runs to the end of the file
            kept.append(data[i:])
            break
        end = i + 2 + int.from_bytes(data[i + 2 : i + 4], "big")
        if not (0xE0 <= marker <= 0xEF or marker == 0xFE):
            kept.append(data[i:end])
        i = end
    return b"".join(kept)


def facts_differ(pixel_only: pd.DataFrame) -> pd.DataFrame:
    """One row per group: True where the copies disagree on a fact."""
    facts = pd.DataFrame([jpeg_facts(i) for i in pixel_only["image_id"]])
    facts["pixel_sha256"] = pixel_only["pixel_sha256"].to_numpy()
    columns = [c for c in facts.columns if c not in ("image_id", "pixel_sha256")]
    return facts.groupby("pixel_sha256")[columns].agg(
        lambda s: s.astype(str).nunique() > 1
    )


def byte_differences(pixel_only: pd.DataFrame) -> pd.DataFrame:
    """One row per copy: where it differs from its group's first image."""
    rows = []
    for h, g in pixel_only.groupby("pixel_sha256", sort=False):
        ref = g.iloc[0]
        a = np.frombuffer((IMAGES / f"{ref['image_id']}.jpg").read_bytes(), np.uint8)
        span = app1_span(a.tobytes())
        ref_image_part = image_part(a.tobytes())
        for _, copy in g.iloc[1:].iterrows():
            if copy["file_sha256"] == ref["file_sha256"]:
                rows.append(
                    {
                        "pixel_sha256": h[:8],
                        "reference": ref["image_id"],
                        "image_id": copy["image_id"],
                        "same_size": True,
                        "n_diff_bytes": 0,  # byte-identical to the reference
                    }
                )
                continue
            b = np.frombuffer(
                (IMAGES / f"{copy['image_id']}.jpg").read_bytes(), np.uint8
            )
            row = {
                "pixel_sha256": h[:8],
                "reference": ref["image_id"],
                "image_id": copy["image_id"],
                "same_size": len(a) == len(b),
                "same_image_part": image_part(b.tobytes()) == ref_image_part,
                "exif_tags_differ": exif_tags_differ(
                    IMAGES / f"{ref['image_id']}.jpg",
                    IMAGES / f"{copy['image_id']}.jpg",
                ),
            }
            if row["same_size"]:
                diff = np.flatnonzero(a != b)
                row |= {
                    "n_diff_bytes": len(diff),
                    "first": int(diff[0]),
                    "last": int(diff[-1]),
                    "all_in_exif": span is not None
                    and bool(((diff >= span[0]) & (diff < span[1])).all()),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def exif_tags(path: Path) -> dict[str, object]:
    """EXIF tags of the main, photo-details (Exif) and GPS groups, by name."""
    with Image.open(path) as im:
        exif = im.getexif()
        main = dict(exif) | dict(exif.get_ifd(ExifTags.IFD.Exif))
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    names = {
        ExifTags.TAGS.get(k, hex(k)): v
        for k, v in main.items()
        if k not in POINTER_TAGS
    }
    names |= {ExifTags.GPSTAGS.get(k, f"GPS{k:#x}"): v for k, v in gps.items()}
    return names


def exif_tags_differ(path_a: Path, path_b: Path) -> str:
    """Names of the EXIF tags whose value differs, or that only one file has."""
    a, b = exif_tags(path_a), exif_tags(path_b)
    return " ".join(sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k)))


def main() -> None:
    review = pd.read_csv(REVIEW, keep_default_na=False)
    pixel_only = review.groupby("pixel_sha256").filter(
        lambda g: g["file_sha256"].nunique() > 1
    )
    print(f"{pixel_only['pixel_sha256'].nunique()} groups with different bytes")

    print("\n== Groups where each fact differs between copies")
    print(facts_differ(pixel_only).sum().to_string())

    where = byte_differences(pixel_only)
    print("\n== Where the bytes differ (each copy vs its group's first image)")
    print(where.to_string(index=False))

    tags = where["exif_tags_differ"].str.split().explode().value_counts()
    print("\n== EXIF tags that differ (number of pairs)")
    print(tags.to_string())
    no_tag = (where["exif_tags_differ"] == "") & (where["n_diff_bytes"] != 0)
    print(f"Pairs with different bytes but no differing tag: {no_tag.sum()}")

    compared = where.dropna(subset=["exif_tags_differ"])  # skip byte-identical copies
    same = compared["same_image_part"].astype(bool).sum()
    print(f"\nPairs with identical tables and image data: {same} of {len(compared)}")


if __name__ == "__main__":
    main()
