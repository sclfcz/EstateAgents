"""下载电影剧情 CSV。检索只使用 Plot，不使用额外摘要列。"""

from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data" / "wiki_movie_plots_deduped_with_summaries.csv"
URL = (
    "https://huggingface.co/datasets/vishnupriyavr/"
    "wiki-movie-plots-with-summaries/resolve/main/"
    "wiki_movie_plots_deduped_with_summaries.csv"
)


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size > 1_000_000:
        print(f"already exists: {TARGET}")
        return
    print(f"downloading {URL}")
    urlretrieve(URL, TARGET)
    print(f"saved {TARGET} ({TARGET.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
