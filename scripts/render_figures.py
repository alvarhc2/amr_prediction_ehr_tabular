"""Draw the windowing figure and a short animation for the README.

The geometry matches the paper. Panels (a)–(c) are 4-year training windows
placed 0, 2 and 4 years before a 1-month test. Panel (d) grows from 2004
and stops 48 hours before the test month. Model numbers in the paper also
depend on which past-resistance features are kept; this figure shows the
windows only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "figures"
TRAIN = "#1F6F8B"
TEST = "#E07A3D"
MARGIN = "#F2C14E"
INK = "#1C2833"
MUTED = "#5C6B73"


def _bar(ax, start, end, y, color, label=None):
    patch = FancyBboxPatch(
        (start, y - 0.28),
        max(end - start, 0.04),
        0.56,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0,
        facecolor=color,
        mutation_aspect=0.3,
    )
    ax.add_patch(patch)
    if label:
        ax.text((start + end) / 2, y, label, ha="center", va="center", color="white", fontsize=8)


def _panel(ax, year, mode):
    ax.set_xlim(2004, 2019.6)
    ax.set_ylim(-0.15, 1.15)
    ax.axhline(0.5, color="#E6EEF2", linewidth=6, zorder=0)
    ax.set_yticks([])
    ax.set_xticks([2004, 2008, 2012, 2016, 2019])
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    test_start = year
    test_end = year + 1 / 12
    if mode == "sliding":
        _bar(ax, year - 4, year, 0.5, TRAIN, "train")
    elif mode == "gap2":
        _bar(ax, year - 6, year - 2, 0.5, TRAIN, "train")
    elif mode == "gap4":
        _bar(ax, year - 8, year - 4, 0.5, TRAIN, "train")
    else:
        _bar(ax, 2004.6, year - 0.03, 0.5, TRAIN, "train")
        ax.scatter([year - 0.015], [0.5], s=22, color=MARGIN, zorder=5)
    _bar(ax, test_start, test_end + 0.05, 0.5, TEST)
    ax.text(test_end + 0.12, 0.5, "test", va="center", ha="left", color=TEST, fontsize=8)


def render_static() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10.4, 5.4), dpi=160)
    titles = [
        "(a)  Sliding window, next to the test month          models M1, M4, M7",
        "(b)  Sliding window, two years before the test       models M2, M5, M8",
        "(c)  Sliding window, four years before the test      models M3, M6, M9",
        "(d)  Incremental window, from the first culture      models M10 to M14",
    ]
    modes = ["sliding", "gap2", "gap4", "incremental"]
    for ax, title, mode in zip(axes, titles, modes):
        _panel(ax, 2016.0, mode)
        ax.set_title(title, loc="left", color=INK, fontsize=10, pad=2)
    fig.suptitle(
        "How a culture in 2016 is predicted from the past",
        fontsize=14,
        color=INK,
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.02,
        "The test is always one month (orange). Training never uses the future. The gold gap in (d) is the 48 hours the antibiogram still needs.",
        color=MUTED,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.90))
    fig.savefig(OUTPUT / "windows.png", facecolor="white")
    plt.close(fig)


def render_gif() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = []
    years = [2016 + step * 0.25 for step in range(10)]
    for year in years:
        fig, axes = plt.subplots(2, 1, figsize=(10.2, 2.8), dpi=140)
        _panel(axes[0], year, "sliding")
        _panel(axes[1], year, "incremental")
        axes[0].set_title("4-year window, moving with the test month", loc="left", color=INK, fontsize=11)
        axes[1].set_title("Incremental window, growing from 2004", loc="left", color=INK, fontsize=11)
        fig.tight_layout()
        path = OUTPUT / "_frame.png"
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        frames.append(Image.open(path).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(
        OUTPUT / "windows.gif",
        save_all=True,
        append_images=frames[1:],
        duration=450,
        loop=0,
        disposal=2,
    )
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    render_static()
    render_gif()
    print(f"Wrote {OUTPUT / 'windows.png'} and {OUTPUT / 'windows.gif'}")
