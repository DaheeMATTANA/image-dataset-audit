# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# %%
ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
IMAGES = ROOT / "data" / "food-101" / "images"
OUT = ROOT / "scratch" / "toy_jpeg"
OUT.mkdir(parents=True, exist_ok=True)

src = next(IMAGES.glob("apple_pie/*.jpg"))
img = Image.open(src).convert("RGB")
print(src.name, round(src.stat().st_size / 1024, 1), "KB")

# %%
arr = np.asarray(img)
print(arr.shape, arr.dtype, arr.min(), arr.max())

# %%
for q in [95, 50, 10]:
    img.save(OUT / f"q{q}.jpg", quality=q)
img.save(OUT / "copy.png")


# %%
def measure(path: Path) -> dict:
    """
    File size in KB and mean absolute pixel error vs. the original array
    """
    size_kb = path.stat().st_size / 1024
    reloaded = np.asarray(Image.open(path).convert("RGB"))
    error = np.abs(arr.astype(np.int16) - reloaded.astype(np.int16)).mean()
    return {
        "file": path.name,
        "size_kb": round(size_kb, 1),
        "mean_abs_error": round(float(error), 2),
    }


# %%
files = [src, OUT / "q95.jpg", OUT / "q50.jpg", OUT / "q10.jpg", OUT / "copy.png"]
results = pd.DataFrame([measure(p) for p in files])
results
# %%
current = img
rows_95 = []

for i in range(1, 11):
    path = OUT / f"q95_gen{i:02d}.jpg"
    current.save(path, quality=95)
    current = Image.open(path).convert("RGB")
    rows_95.append({"generation": i, **measure(path)})

iterate_95 = pd.DataFrame(rows_95)
iterate_95


# %%
def generation_loss(start: Image.Image, quality: int, n: int = 10) -> pd.DataFrame:
    """
    Re-save an image n times at a fixed JPEG quality, measuring error vs. the original
    """
    current = start
    rows = []

    for i in range(1, n + 1):
        path = OUT / f"q{quality}_gen{i:02d}.jpg"
        current.save(path, quality=quality)
        current = Image.open(path).convert("RGB")
        rows.append({"quality": quality, "generation": i, **measure(path)})
    return pd.DataFrame(rows)


iterate_95 = generation_loss(img, 95)
iterate_50 = generation_loss(img, 50)
pd.concat([iterate_95, iterate_50])

# %%
qualities = [95, 80]

rows = []
for q in qualities:
    path = OUT / f"q{q}.jpg"
    img.save(path, quality=q)
    rows.append({"quality": q, **measure(path)})

single = pd.DataFrame(rows)
single

# %%
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
# strict=False: the palette may be longer than the list of qualities
COLORS = dict(zip(qualities, PALETTE, strict=False))
gl = pd.concat([generation_loss(img, q) for q in qualities])

INK, GRID = "#52514e", "#e4e3df"

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
panels = [
    ("mean_abs_error", "Mean absolute pixel error vs original", "error (0–255 scale)"),
    ("size_kb", "File size", "KB"),
]
for ax, (col, title, ylabel) in zip(axes, panels, strict=True):
    for q, color in COLORS.items():
        d = gl[gl["quality"] == q]
        ax.plot(
            d["generation"],
            d[col],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"q={q}",
        )
        last = d.iloc[-1]  # label each line at its end
        ax.annotate(
            f"q={q}: {last[col]:g}",
            (last["generation"], last[col]),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=9,
        )
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xlabel("re-save generation", color=INK)
    ax.set_ylim(0, gl[col].max() * 1.2)
    ax.set_xlim(0.5, 12.5)  # room for the end labels
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9)
fig.suptitle(
    "Generation loss: re-saving the same JPEG 10 times",
    x=0.01,
    ha="left",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(ROOT / "reports" / "jpeg_quality_iteration_experiment.png", dpi=150)
plt.show()
