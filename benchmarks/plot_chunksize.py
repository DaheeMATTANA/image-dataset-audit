"""Plot speedup against workers, one line per chunksize, with the batch-count limit."""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reports" / "benchmark_chunksize.csv"
OUT = ROOT / "reports" / "benchmark_chunksize_plot.png"

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # one fixed color per chunksize (up to 3)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def batch_limit(n_items: int, chunksize: int, workers: int) -> float:
    """Best possible speedup when work comes in equal batches.

    Total batches divided by the batches on the busiest worker.
    """
    n_batches = math.ceil(n_items / chunksize)
    return n_batches / math.ceil(n_batches / workers)


def main() -> None:
    df = pd.read_csv(CSV)
    size = df["sample"].max()
    df = df[df["sample"] == size]
    base = df[df["workers"] == 1]
    parallel = df[df["workers"] > 1]
    chunks = sorted(parallel["chunksize"].dropna().astype(int).unique())
    if len(chunks) > len(COLORS):
        sys.exit(f"{len(chunks)} chunk sizes but only {len(COLORS)} colors defined")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    workers = sorted(df["workers"].unique())
    ax.plot(workers, workers, color=MUTED, lw=1, ls="--", label="ideal (linear)")

    for color, cs in zip(COLORS, chunks, strict=False):
        d = pd.concat([base, parallel[parallel["chunksize"] == cs]]).sort_values(
            "workers"
        )
        ax.plot(
            d["workers"],
            d["speedup"],
            color=color,
            lw=2,
            marker="o",
            ms=5,
            label=f"chunksize {cs}",
        )
        # Draw the cap only when there are fewer batches than workers
        if math.ceil(size / cs) < max(workers):
            cap = [batch_limit(size, cs, w) for w in d["workers"]]
            ax.plot(
                d["workers"],
                cap,
                color=color,
                lw=1,
                ls=":",
                label=f"batch limit, chunksize {cs}",
            )

    ax.set_xticks(workers)
    ax.set_ylim(0, None)
    ax.set_xlabel("workers", color=MUTED)
    ax.set_ylabel("speedup vs 1 worker", color=MUTED)
    ax.set_title(
        f"Speedup by chunksize ({size:,} images)",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
