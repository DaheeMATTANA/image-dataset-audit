"""Why do pixel-identical copies differ in bytes?

Takes the pixel-duplicate groups whose files are not byte-identical (more than one
file_sha256 in annotations/exact_duplicate_review.csv) and compares each copy with
its group's first image:

1. Category: identical, metadata (only metadata blocks differ), coding (same
   quantization tables, other decoder blocks differ) or quantization (quantization
   tables differ). Plus whether the file sizes differ, whether the orientation
   tag differs as stored, and whether the copies would display rotated
   differently in software that applies the tag.
2. Reason: which metadata blocks differ and, inside EXIF, which tag values differ
   (or "unused bytes" when the only differing bytes are zero in one copy).
3. Details: which JPEG facts differ per group, and the table of compared pairs.

Each file is read and parsed once (file_info); the comparisons only use the
results, so the cost grows with the number of files, not with repeated reads.

    uv run python experiments/duplicate_bytes_experiment.py
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ExifTags, Image, JpegImagePlugin

from image_dataset_audit.dataset import IMAGES, REPORTS, ROOT

REVIEW = ROOT / "annotations" / "exact_duplicate_review.csv"
CATEGORIES = ["identical", "metadata", "coding", "quantization"]
METADATA_MARKERS = set(range(0xE0, 0xF0)) | {0xFE}  # APP0-APP15 and COM blocks
QUANT_TABLES = 0xDB  # DQT block
ORIENTATION = 274  # EXIF tag number
# Tags that hold the position of another EXIF block, not a value
POINTER_TAGS = {ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo, ExifTags.IFD.Interop}


def jpeg_blocks(data: bytes) -> list[tuple[int, bytes]]:
    """(marker, bytes) of each block, in file order.

    Walks the blocks by their length fields. The scan and image data, or the rest
    of the file if the walk stops early, is kept as the last block (marker 0xDA),
    so nothing is ever dropped.
    """
    blocks, i = [], 2  # skip the start-of-image marker (FF D8)
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xFF:  # filler byte before a marker
            i += 1
            continue
        if marker == 0xDA:  # start of scan: image data runs to the end
            break
        end = i + 2 + int.from_bytes(data[i + 2 : i + 4], "big")
        blocks.append((marker, data[i:end]))
        i = end
    blocks.append((0xDA, data[i:]))
    return blocks


def block_name(marker: int, block: bytes) -> str:
    """Readable name of a metadata block, from its marker and first bytes."""
    payload = block[4:]  # after the marker and the length field
    if marker == 0xE1 and payload.startswith(b"Exif"):
        return "EXIF"
    if marker == 0xE1 and payload.startswith(b"http://ns.adobe.com/xap"):
        return "XMP"
    if marker == 0xE2 and payload.startswith(b"ICC_PROFILE"):
        return "ICC"
    names = {0xE0: "JFIF", 0xED: "Photoshop", 0xEE: "Adobe", 0xFE: "comment"}
    return names.get(marker, f"APP{marker - 0xE0}")


def metadata_blocks(blocks: list[tuple[int, bytes]]) -> dict[str, bytes]:
    """Metadata blocks by name; blocks with the same name are joined."""
    named: dict[str, bytes] = {}
    for marker, block in blocks:
        if marker in METADATA_MARKERS:
            name = block_name(marker, block)
            named[name] = named.get(name, b"") + block
    return named


def tag_names(main: dict, gps: dict) -> dict[str, object]:
    """EXIF tags of the main and photo-details (Exif) groups and GPS, by name."""
    names = {
        ExifTags.TAGS.get(k, hex(k)): v
        for k, v in main.items()
        if k not in POINTER_TAGS
    }
    names |= {ExifTags.GPSTAGS.get(k, f"GPS{k:#x}"): v for k, v in gps.items()}
    return names


def file_info(path: Path) -> dict:
    """Everything the comparisons need from one file, read and parsed once.

    Pillow parses from the bytes already in memory, so the file is read once.
    Its warnings about damaged EXIF are recorded instead of printed.
    """
    data = path.read_bytes()
    blocks = jpeg_blocks(data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Image.open(io.BytesIO(data)) as im:
            exif = im.getexif()
            main = dict(exif) | dict(exif.get_ifd(ExifTags.IFD.Exif))
            gps = dict(exif.get_ifd(ExifTags.IFD.GPSInfo))
            # As Pillow reads it: from EXIF, or from XMP
            orientation_tag = exif.get(ORIENTATION)
            facts = {
                "size_bytes": len(data),
                "exif_bytes": len(im.info.get("exif", b"")),
                "orientation_tag": orientation_tag,
                "icc": "icc_profile" in im.info,
                "progressive": bool(im.info.get("progressive", 0)),
                "comment": im.info.get("comment"),
                "quant": im.quantization,
                "subsampling": JpegImagePlugin.get_sampling(im),
                "info_keys": tuple(sorted(im.info)),
            }
    return {
        "data": data,
        "metadata": metadata_blocks(blocks),
        "decoder": [blk for m, blk in blocks if m not in METADATA_MARKERS],
        # Tables only (after each block's 4-byte header), so one block holding
        # two tables equals two blocks holding one each
        "quant_tables": b"".join(blk[4:] for m, blk in blocks if m == QUANT_TABLES),
        "tags": tag_names(main, gps),
        "orientation_tag": orientation_tag,
        "damaged_exif": bool(caught),
        "facts": facts,
    }


def displayed_orientation(tag: object) -> int:
    """Orientation as a viewer applies it: missing or invalid (not 1-8) = 1."""
    # Damaged EXIF can hold a non-number here
    return tag if isinstance(tag, int) and 1 <= tag <= 8 else 1


def category(a: dict, b: dict) -> str:
    """identical, metadata (cause a), coding (likely b) or quantization (likely c)."""
    if a["data"] == b["data"]:
        return "identical"
    if a["decoder"] == b["decoder"]:
        return "metadata"
    return "coding" if a["quant_tables"] == b["quant_tables"] else "quantization"


def tags_differ(a: dict, b: dict) -> str:
    """Names of the EXIF tags whose value differs, or that only one file has."""
    ta, tb = a["tags"], b["tags"]
    return " ".join(sorted(k for k in ta.keys() | tb.keys() if ta.get(k) != tb.get(k)))


def n_diff_bytes(a: bytes, b: bytes) -> int | None:
    """Number of bytes that differ, or None if the files have different sizes."""
    if len(a) != len(b):
        return None
    x, y = np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)
    return int(np.count_nonzero(x != y))


def unused_bytes(a: bytes, b: bytes) -> bool:
    """True if a and b have the same length and differ only where one has zeros."""
    if len(a) != len(b):
        return False
    x, y = np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)
    diff = x != y
    return bool(diff.any() and ((x[diff] == 0).all() or (y[diff] == 0).all()))


def reason(a: dict, b: dict, tags: str) -> str:
    """Where two files differ: metadata block names and, inside EXIF, why."""
    meta_a, meta_b = a["metadata"], b["metadata"]
    differ = sorted(
        n for n in meta_a.keys() | meta_b.keys() if meta_a.get(n) != meta_b.get(n)
    )
    parts = [n for n in differ if n != "EXIF"]
    if "EXIF" in differ:
        if tags:
            parts.append(f"EXIF: {tags}")
        elif unused_bytes(meta_a.get("EXIF", b""), meta_b.get("EXIF", b"")):
            parts.append("EXIF: unused bytes")
        else:
            parts.append("EXIF: unexplained")
    return "; ".join(parts)


def facts_differ(pixel_only: pd.DataFrame, infos: dict[str, dict]) -> pd.DataFrame:
    """One row per group: True where the copies disagree on a fact."""
    facts = pd.DataFrame([infos[i]["facts"] for i in pixel_only["image_id"]])
    facts["pixel_sha256"] = pixel_only["pixel_sha256"].to_numpy()
    columns = [c for c in facts.columns if c != "pixel_sha256"]
    return facts.groupby("pixel_sha256")[columns].agg(
        lambda s: s.astype(str).nunique(dropna=False) > 1
    )


def compare_pairs(pixel_only: pd.DataFrame, infos: dict[str, dict]) -> pd.DataFrame:
    """One row per copy: how it differs from its group's first image."""
    rows = []
    for h, g in pixel_only.groupby("pixel_sha256", sort=False):
        ref_id = g["image_id"].iloc[0]
        a = infos[ref_id]
        for copy_id in g["image_id"].iloc[1:]:
            b = infos[copy_id]
            same = a["data"] == b["data"]
            tags = "" if same else tags_differ(a, b)
            rows.append(
                {
                    "pixel_sha256": h[:8],
                    "reference": ref_id,
                    "image_id": copy_id,
                    "category": category(a, b),
                    "orientation_tag_differs": a["orientation_tag"]
                    != b["orientation_tag"],
                    "rotation_differs": displayed_orientation(a["orientation_tag"])
                    != displayed_orientation(b["orientation_tag"]),
                    "same_size": len(a["data"]) == len(b["data"]),
                    "n_diff_bytes": n_diff_bytes(a["data"], b["data"]),
                    "exif_tags_differ": tags,
                    "reason": "" if same else reason(a, b, tags),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    review = pd.read_csv(REVIEW, keep_default_na=False)
    pixel_only = review.groupby("pixel_sha256").filter(
        lambda g: g["file_sha256"].nunique() > 1
    )
    infos = {i: file_info(IMAGES / f"{i}.jpg") for i in pixel_only["image_id"]}
    pairs = compare_pairs(pixel_only, infos)
    print(
        f"{pixel_only['pixel_sha256'].nunique()} groups with different bytes, "
        f"{len(pairs)} pairs compared (each copy vs its group's first image)"
    )

    print("\n== Category per pair")
    print(
        pairs["category"].value_counts().reindex(CATEGORIES, fill_value=0).to_string()
    )
    print("\n== Flags (each counted separately; a pair can have several)")
    n = len(pairs)
    damaged = sum(info["damaged_exif"] for info in infos.values())
    print(f"Different file sizes: {(~pairs['same_size']).sum()} of {n} pairs")
    print(
        "Orientation tag differs: "
        f"{pairs['orientation_tag_differs'].sum()} of {n} pairs"
    )
    print(
        f"Display rotated differently: {pairs['rotation_differs'].sum()} of {n} pairs"
    )
    print(f"Damaged EXIF (Pillow warnings): {damaged} of {len(infos)} files")

    print("\n== Different file sizes, per category")
    sizes = pairs.assign(different_size=~pairs["same_size"])
    print(pd.crosstab(sizes["category"], sizes["different_size"]).to_string())

    print("\n== Reason per pair")
    print(pairs["reason"].value_counts(dropna=False).to_string())

    print("\n== Details: groups where each JPEG fact differs between copies")
    print(facts_differ(pixel_only, infos).sum().to_string())

    print("\n== Details: compared pairs")
    print(pairs.to_string(index=False))

    # Whole numbers; empty where the sizes differ
    pairs["n_diff_bytes"] = pairs["n_diff_bytes"].astype("Int64")
    pairs.to_csv(REPORTS / "experiment_duplicate_bytes.csv", index=False)


if __name__ == "__main__":
    main()
