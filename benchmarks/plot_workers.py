"""Plot throughput and efficiency against worker count from benchmark_workers.csv.

One line per sample size. The CSV holds a single chunksize; chunk sizes are
compared by plot_chunksize.py.
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

from image_dataset_audit.dataset import REPORTS

CSV = REPORTS / "benchmark_workers.csv"
OUT = REPORTS / "benchmark_workers_plot.png"

# Hardware boundaries of this machine (i7-13650HX), drawn as reference lines.
BOUNDARIES = {6: "all P-cores", 14: "all physical cores"}

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # one fixed color per sample size
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    # chunksize is empty for the 1-worker baseline (one process ignores it)
    df = pd.read_csv(CSV, dtype={"chunksize": "Int64"})
    timed = sorted(int(c) for c in df["chunksize"].dropna().unique())
    if len(timed) != 1:
        sys.exit(f"Expected one chunksize in {CSV.name}, found {timed}")
    chunksize = timed[0]
    sizes = sorted(df["sample"].unique())
    if len(sizes) > len(COLORS):
        sys.exit(f"{len(sizes)} sample sizes but only {len(COLORS)} colors defined")

    fig, (ax_tp, ax_eff) = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)

    for color, size in zip(COLORS, sizes, strict=False):
        d = df[df["sample"] == size].sort_values("workers")
        label = f"{size:,} images"
        ax_tp.plot(
            d["workers"],
            d["images_per_s"],
            color=color,
            lw=2,
            marker="o",
            ms=5,
            label=label,
        )
        ax_eff.plot(
            d["workers"],
            d["efficiency"] * 100,
            color=color,
            lw=2,
            marker="o",
            ms=5,
            label=label,
        )

    # Ideal linear scaling from the 1-worker throughput of the largest sample.
    largest = df[df["sample"] == sizes[-1]]
    one = largest.loc[largest["workers"] == 1, "images_per_s"].iloc[0]
    xs = sorted(df["workers"].unique())
    ax_tp.plot(
        xs,
        [one * x for x in xs],
        color=MUTED,
        lw=1,
        ls="--",
        label="ideal (linear)",
    )

    for ax in (ax_tp, ax_eff):
        for x, text in BOUNDARIES.items():
            ax.axvline(x, color=GRID, lw=1.5, zorder=0)
            ax.annotate(
                text,
                (x, 1),
                xycoords=("data", "axes fraction"),
                xytext=(4, -4),
                textcoords="offset points",
                va="top",
                fontsize=8,
                color=MUTED,
            )
        ax.set_xticks(sorted(df["workers"].unique()))
        ax.set_xlabel("workers", color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    ax_tp.set_title("Throughput", loc="left", fontsize=11, color=INK)
    ax_tp.set_ylabel("images per second", color=MUTED)
    ax_tp.set_ylim(0, None)
    ax_eff.set_title(
        "Efficiency (speedup / workers)", loc="left", fontsize=11, color=INK
    )
    ax_eff.set_ylabel("%", color=MUTED)
    ax_eff.set_ylim(0, 105)

    fig.suptitle(
        f"check_integrity scaling with worker processes (chunksize {chunksize})",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    # Legend on its own row under the title, so a long title can't run into it
    handles, labels = ax_tp.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.92),
        ncol=len(labels),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(OUT, dpi=150)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
