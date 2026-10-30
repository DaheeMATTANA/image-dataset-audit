"""
Plot the image profile written by src/profile_images.py (run that first).

Input:  outputs/profile_images.csv
Output: reports/profile_short_side.png, profile_file_size.png,
        profile_width_height.png, profile_mode.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import StrMethodFormatter

from dataset_utils import OUTPUTS, REPORTS

# Same colours as experiments/jpeg_png_experiment.py
SPLIT_COLORS = {"train": "#2a78d6", "test": "#eb6834"}
BAR = "#2a78d6"
INK, GRID, SURFACE = "#52514e", "#e4e3df", "#ffffff"
# One hue, light -> dark, for "how many images" (sequential)
COUNT_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)
THOUSANDS = StrMethodFormatter("{x:,.0f}")
KB = 1000  # decimal units, same as reports/profile_summary.csv


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str, grid: str = "y"):
    """Shared look: left title, light grid behind the data, no top/right frame."""
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)


def save(fig: plt.Figure, path: Path) -> None:
    """Write the figure as PNG and free its memory."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPORTS.parent)}")


def plot_short_side(df: pd.DataFrame, path: Path) -> None:
    """Histogram of the short side, with the two most common sizes labelled."""
    short = df[["width", "height"]].min(axis=1)
    long_512_pct = (df[["width", "height"]].max(axis=1) == 512).mean() * 100
    bins = np.arange(short.min() // 8 * 8, 512 + 9, 8)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(short, bins=bins, color=BAR, rwidth=0.85)
    for value in short.value_counts().head(2).index:
        n = int((short == value).sum())
        ax.annotate(
            f"{value} px: {n:,} ({n / len(short):.0%})",
            xy=(value + 4, n),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=9,
            color=INK,
        )
    ax.yaxis.set_major_formatter(THOUSANDS)
    style_axes(
        ax,
        f"Short side per image (long side is 512 px for {long_512_pct:.1f}%)",
        "short side (px, 8 px bins)",
        "images",
    )
    save(fig, path)


def plot_file_size(df: pd.DataFrame, path: Path) -> None:
    """File-size distribution per split, as % of the split so sizes are comparable."""
    kb = df["size_bytes"] / KB
    x_max = kb.quantile(0.999)
    bins = np.arange(0, x_max + 2, 2)

    fig, ax = plt.subplots(figsize=(8, 4))
    for split, color in SPLIT_COLORS.items():
        s = kb[df["split"] == split]
        ax.hist(
            s,
            bins=bins,
            weights=np.full(len(s), 100 / len(s)),
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{split}: {len(s):,} images, median {s.median():.1f} KB",
        )
    ax.set_xlim(0, x_max)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    style_axes(
        ax,
        "File size per split",
        f"file size (KB, 2 KB bins; cut at 99.9th percentile, max {kb.max():.0f} KB)",
        "% of images in split",
    )
    save(fig, path)


def plot_width_height(df: pd.DataFrame, path: Path) -> None:
    """One dot per distinct width x height, coloured by how many images have it."""
    sizes = df.groupby(["width", "height"]).size().reset_index(name="n")
    sizes = sizes.sort_values("n")  # most common drawn last, on top
    n_land = int((df["width"] > df["height"]).sum())
    n_port = int((df["width"] < df["height"]).sum())
    n_square = int((df["width"] == df["height"]).sum())

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 512], [0, 512], color=GRID, linewidth=1, linestyle="--", zorder=0)
    dots = ax.scatter(
        sizes["width"],
        sizes["height"],
        c=sizes["n"],
        cmap=COUNT_CMAP,
        norm=LogNorm(),
        s=22,
        edgecolors=SURFACE,
        linewidths=0.5,
    )
    cbar = fig.colorbar(dots, ax=ax, shrink=0.8)
    cbar.set_label("images with this size (log scale)", color=INK)
    cbar.ax.tick_params(colors=INK, labelsize=9)
    cbar.outline.set_visible(False)
    ax.text(
        0.03,
        0.03,
        f"landscape {n_land:,}\nportrait {n_port:,}\nsquare {n_square:,}",
        transform=ax.transAxes,
        fontsize=9,
        color=INK,
    )
    ax.set_aspect("equal")
    ax.set_xlim(0, 530)
    ax.set_ylim(0, 530)
    style_axes(
        ax,
        f"Image sizes: {len(sizes)} distinct width × height",
        "width (px)",
        "height (px)",
        grid="both",
    )
    save(fig, path)


def plot_mode(df: pd.DataFrame, path: Path) -> None:
    """Image count per colour mode and split; counts printed since tiny bars vanish."""
    counts = pd.crosstab(df["mode"], df["split"]).reindex(
        columns=list(SPLIT_COLORS), fill_value=0
    )
    modes = counts.sum(axis=1).sort_values(ascending=False).index
    rows = [
        (f"{mode} · {split}", int(counts.loc[mode, split]))
        for mode in modes
        for split in SPLIT_COLORS
    ]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.2))
    labels, values = zip(*rows, strict=True)
    y = np.arange(len(rows))[::-1]  # first row at the top
    ax.barh(y, values, color=BAR, height=0.6)
    for yi, value in zip(y, values, strict=True):
        ax.text(value, yi, f"  {value:,}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.set_major_formatter(THOUSANDS)
    style_axes(
        ax,
        "Colour mode per split (RGB = colour, L = grayscale)",
        "images",
        "",
        grid="x",
    )
    save(fig, path)


def main() -> None:
    src = OUTPUTS / "profile_images.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found: run src/profile_images.py first")
    df = pd.read_csv(src)
    df = df[df["error"].isna()]

    plot_short_side(df, REPORTS / "profile_short_side.png")
    plot_file_size(df, REPORTS / "profile_file_size.png")
    plot_width_height(df, REPORTS / "profile_width_height.png")
    plot_mode(df, REPORTS / "profile_mode.png")


if __name__ == "__main__":
    main()
