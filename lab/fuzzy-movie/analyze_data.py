"""统计片库分布，并导出实验报告用的图。"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "data" / "wiki_movie_plots_deduped.csv",
    ROOT / "data" / "wiki_movie_plots_deduped_with_summaries.csv",
]
OUT = ROOT / "report" / "figures"


def main() -> None:
    path = next(item for item in DATA_CANDIDATES if item.exists())
    csv.field_size_limit(10**7)
    origins = Counter()
    decades = Counter()
    lengths = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            origins[row.get("Origin/Ethnicity") or "未知"] += 1
            year = row.get("Release Year") or ""
            decade = f"{year[:3]}0s" if len(year) >= 3 and year[:3].isdigit() else "未知"
            decades[decade] += 1
            lengths.append(len((row.get("Plot") or "").split()))

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    top = origins.most_common(12)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([name for name, _ in reversed(top)], [count for _, count in reversed(top)], color="#9a3412")
    ax.set_xlabel("Movies")
    ax.set_title("Top origins in the plot corpus")
    fig.tight_layout()
    fig.savefig(OUT / "origins.png", dpi=140)
    plt.close(fig)

    labels = sorted(decades)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(labels, [decades[label] for label in labels], color="#1c1915")
    ax.set_title("Movies by decade")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "decades.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(lengths, bins=40, color="#c2410c")
    ax.set_xlabel("Plot length (words)")
    ax.set_title("Plot length distribution")
    fig.tight_layout()
    fig.savefig(OUT / "lengths.png", dpi=140)
    plt.close(fig)
    print("movies", sum(origins.values()))
    print("median words", sorted(lengths)[len(lengths) // 2])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
