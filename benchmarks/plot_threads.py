"""Plot throughput against worker count for threads vs processes.

Reads reports/benchmark_threads.csv (written by bench_threads.py). With one worker
both kinds run the same serial loop, so both lines start from that row.
"""

import matplotlib.pyplot as plt
import pandas as pd

from image_dataset_audit.dataset import REPORTS

CSV = REPORTS / "benchmark_threads.csv"
OUT = REPORTS / "benchmark_threads_plot.png"

# Hardware boundaries of this machine (i7-13650HX), drawn as reference lines.
BOUNDARIES = {6: "all P-cores", 14: "all physical cores"}

# Same colors as plot_workers.py; marker shape is a second cue besides color.
STYLES = {
    "process": {"color": "#2a78d6", "marker": "o", "label": "processes"},
    "thread": {"color": "#eb6834", "marker": "s", "label": "threads"},
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    df = pd.read_csv(CSV)
    serial = df[df["kind"] == "serial"]
    sample = int(df["sample"].iloc[0])
    xs = sorted(df["workers"].unique())

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for kind, style in STYLES.items():
        d = pd.concat([serial, df[df["kind"] == kind]]).sort_values("workers")
        ax.plot(
            d["workers"],
            d["images_per_s"],
            color=style["color"],
            marker=style["marker"],
            lw=2,
            ms=6,
            label=style["label"],
        )
        # Direct label at the line's end, in text color (the marker carries identity).
        last = d.iloc[-1]
        ax.annotate(
            f"{style['label']}  {last['images_per_s']:,.0f}",
            (last["workers"], last["images_per_s"]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )

    # Ideal linear scaling from the serial loop; it leaves the top of the axes
    # after a few workers, so the measured lines keep the vertical space.
    one = serial["images_per_s"].iloc[0]
    ax.plot(
        xs,
        [one * x for x in xs],
        color=MUTED,
        lw=1,
        ls="--",
        label="ideal (linear)",
    )

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

    ax.set_xticks(xs)
    ax.set_xlim(0, max(xs) + 4)  # room for the end labels
    ax.set_ylim(0, df["images_per_s"].max() * 1.2)
    ax.set_xlabel("workers (threads or processes)", color=MUTED)
    ax.set_ylabel("images per second", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle(
        f"check_integrity: threads vs processes ({sample:,} images)",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    handles, labels = ax.get_legend_handles_labels()
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
