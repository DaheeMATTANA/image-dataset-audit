# %%
import hashlib
import random
import shutil
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

# %%
ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
IMAGES = ROOT / "data" / "food-101" / "images"

# %%
src = next(IMAGES.glob("apple_pie/*.jpg"))
print(src.name, src.stat().st_size, "bytes")


# %%
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so memory use stays constant."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


print(sha256_file(src))

# %%
# same file, same hash
whole_file = hashlib.sha256(src.read_bytes()).hexdigest()
with open(src, "rb") as f:
    builtin = hashlib.file_digest(f, "sha256").hexdigest()

print(sha256_file(src) == whole_file == builtin)

# %%
# 1. Copy the file, same hash?
OUT = ROOT / "scratch" / "hashing"
OUT.mkdir(parents=True, exist_ok=True)

copy = OUT / "copy.jpg"
shutil.copy2(src, copy)

print(sha256_file(src))
print(sha256_file(copy))
print("same hash: ", sha256_file(src) == sha256_file(copy))

# %%
# 2. flip one bit, how many output bits change? (Avalanche effect)

data = bytearray(src.read_bytes())
pos = len(data) // 2  # a byte in the middle of the file
data[pos] ^= 0x01  # flip its lowest bit: the smallest possible change

flipped = OUT / "flipped.jpg"
flipped.write_bytes(data)


def bits_changed(h1: str, h2: str) -> int:
    """Number of differing bits between two hex digests."""
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


h_orig, h_flip = sha256_file(src), sha256_file(flipped)
print(h_orig)
print(h_flip)
print("bits changed: ", bits_changed(h_orig, h_flip), "of 256")

# %%
# 3. repeat 1000 times
original = src.read_bytes()
h0 = hashlib.sha256(original).hexdigest()
rng = random.Random(0)  # fixed seed: reproducible
counts = []

for _ in range(1000):
    d = bytearray(original)
    d[rng.randrange(len(d))] ^= 1 << rng.randrange(8)
    counts.append(bits_changed(h0, hashlib.sha256(d).hexdigest()))

print(
    "min",
    min(counts),
    "mean",
    round(statistics.mean(counts), 1),
    "max",
    max(counts),
    "std",
    round(statistics.pstdev(counts), 1),
)


# %%
# pixel hash
def sha256_pixels(path: Path) -> str:
    """SHA-256 of the decoded RGB pixels (shape included)."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))  # decode: shape (H, W, 3), uint8
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())  # raw pixel values, one byte each
    return h.hexdigest()


# %%
resaved = OUT / "resaved_q95.jpg"
with Image.open(src) as im:
    im.save(resaved, quality=95)

tagged = OUT / "tagged.jpg"
tagged.write_bytes(src.read_bytes() + b"\nsource: reposted copy")

# %%
# compare them
ref_file, ref_pixels = sha256_file(src), sha256_pixels(src)
rows = []
for name, p in [
    ("original", src),
    ("exact copy", copy),
    ("extra bytes appended", tagged),
    ("re-saved q=95", resaved),
]:
    f, px = sha256_file(p), sha256_pixels(p)
    rows.append(
        {
            "version": name,
            "file_hash": f[:12],
            "pixel_hash": px[:12],
            "same_bytes": f == ref_file,
            "same_pixels": px == ref_pixels,
        }
    )

pd.DataFrame(rows)

# %%
# bad copies

BAD = ROOT / "scratch" / "bad_copies"
BAD.mkdir(parents=True, exist_ok=True)

# check this setting before creating bad copies (True would hide truncation)
print("LOAD_TRUNCATED_IMAGES: ", ImageFile.LOAD_TRUNCATED_IMAGES)
print(src.name, len(original), "bytes")

cases = {"intact (control)": src}

p = BAD / "truncated_50.jpg"
p.write_bytes(original[: len(original) // 2])  # keep the first half
cases["truncated 50%"] = p

p = BAD / "truncated_99.jpg"
p.write_bytes(original[: int(len(original) * 0.99)])  # lose only the last 1%
cases["truncated 99%"] = p

p = BAD / "zero_bytes.jpg"
p.write_bytes(b"")  # empty file
cases["zero bytes"] = p

d = bytearray(original)
mid = len(d) // 2
rng = random.Random(0)
# 1000 random bytes mid-file
d[mid : mid + 1000] = bytes(rng.randrange(256) for _ in range(1000))
p = BAD / "random_mid.jpg"
p.write_bytes(d)
cases["random bytes mid-file"] = p

p = BAD / "png_renamed.jpg"
with Image.open(src) as im:
    im.save(p, format="PNG")
cases["PNG named .jpg"] = p


# %%
def check(path: Path, level: str) -> str:
    """Run one check level on a fresh open; return "ok (FORMAT)" or the error."""
    try:
        with Image.open(path) as im:
            fmt = im.format
            if level == "verify":
                im.verify()
            elif level == "load":
                im.load()
        return f"ok ({fmt})"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


levels = ["open", "verify", "load"]
results = pd.DataFrame(
    [
        {"case": name, **{lvl: check(p, lvl) for lvl in levels}}
        for name, p in cases.items()
    ]
)
results
# %%
