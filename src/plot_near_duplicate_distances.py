"""Plot pairs per Hamming distance per hash (near_duplicate_distance_counts.csv).

Log scale, because counts run from tens to ~10^9. Left: all distances 0-64.
Right: distances 0-20, where the radius is chosen. Distances with 0 pairs are
left out (log(0) is undefined): pHash only has even distances.

    uv run python src/plot_near_duplicate_distances.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from find_near_duplicate_candidates import RADIUS
from image_dataset_audit.dataset import REPORTS

CSV = REPORTS / "near_duplicate_distance_counts.csv"
OUT = REPORTS / "near_duplicate_distance_counts_plot.png"

# One fixed color and marker per hash (marker: identity without color alone)
STYLE = {
    "ahash": ("#2a78d6", "o", "aHash"),
    "dhash": ("#eb6834", "s", "dHash"),
    "phash": ("#1baf7a", "^", "pHash"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ZOOM = 20  # last distance shown in the right panel


def main() -> None:
    df = pd.read_csv(CSV)
    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.6))

    for name, (color, marker, label) in STYLE.items():
        d = df[df[name] > 0]  # log scale: drop distances with no pairs
        for ax, part in ((ax_all, d), (ax_zoom, d[d["distance"] <= ZOOM])):
            ax.plot(
                part["distance"],
                part[name],
                color=color,
                lw=2,
                marker=marker,
                ms=4 if ax is ax_all else 6,
                label=label,
            )

    radii = sorted(set(RADIUS.values()))
    radius_text = ", ".join(
        f"{STYLE[h][2]} {r}" for h, r in RADIUS.items()
    )  # e.g. "dHash 12, pHash 12"
    for ax in (ax_all, ax_zoom):
        for r in radii:
            ax.axvline(r, color=MUTED, lw=1, ls="--", zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("Hamming distance (bits differing out of 64)", color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    ax_zoom.annotate(
        f"radius ({radius_text})",
        (radii[-1], 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    ax_all.set_title("All distances", loc="left", fontsize=11, color=INK)
    ax_all.set_ylabel("image pairs (log scale)", color=MUTED)
    ax_all.set_xlim(0, 64)
    ax_zoom.set_title(f"Distances 0-{ZOOM}", loc="left", fontsize=11, color=INK)
    ax_zoom.set_xlim(-0.5, ZOOM + 0.5)
    ax_zoom.set_xticks(range(0, ZOOM + 1, 2))

    n_pairs = int(df["dhash"].sum())
    fig.suptitle(
        f"Image pairs per perceptual-hash distance ({n_pairs:,} pairs per hash)",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    # Legend on its own row under the title
    handles, labels = ax_all.get_legend_handles_labels()
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
