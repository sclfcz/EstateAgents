"""把每部电影的剧情切成短句，编码成多语言句向量。

检索只用 Plot，不用 PlotSummary。短句保留开头和结尾的细节，
例如《盗梦空间》最后的陀螺不会在截断时被丢掉。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from textutil import plot_pieces

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "data" / "wiki_movie_plots_deduped.csv",
    ROOT / "data" / "wiki_movie_plots_deduped_with_summaries.csv",
]
INDEX_DIR = ROOT / "index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def find_csv() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("未找到电影剧情 CSV，请先运行 python download_data.py")


def load_rows(path: Path) -> list[dict]:
    csv.field_size_limit(10**7)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preview(plot: str, limit: int = 720) -> str:
    text = " ".join((plot or "").split())
    if len(text) <= limit:
        return text
    head = text[:460].rsplit(" ", 1)[0]
    tail = text[-220:].split(" ", 1)[-1]
    return f"{head} … {tail}"


def main() -> None:
    rows = load_rows(find_csv())
    print(f"movies {len(rows)}", flush=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    owners: list[int] = []
    with (INDEX_DIR / "movies.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            pieces = plot_pieces(row.get("Plot") or "")
            owners.extend([index] * len(pieces))
            record = {
                "id": index,
                "year": row.get("Release Year") or "",
                "title": row.get("Title") or "",
                "origin": row.get("Origin/Ethnicity") or "",
                "director": row.get("Director") or "",
                "cast": " ".join((row.get("Cast") or "").split())[:160],
                "genre": row.get("Genre") or "",
                "wiki": row.get("Wiki Page") or "",
                "plot_preview": preview(row.get("Plot") or ""),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    owner_array = np.asarray(owners, dtype=np.int32)
    np.save(INDEX_DIR / "sentence_owner.npy", owner_array)
    print(f"sentences {len(owner_array)}", flush=True)

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_embedding_dimension()
    embeddings = np.lib.format.open_memmap(
        INDEX_DIR / "sentence_emb.f16.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(owner_array), dim),
    )
    centroids = np.zeros((len(rows), dim), dtype=np.float32)
    counts = np.zeros(len(rows), dtype=np.int32)

    cursor = 0
    batch_movies = 400
    for start in range(0, len(rows), batch_movies):
        chunk_rows = rows[start : start + batch_movies]
        texts: list[str] = []
        local_owners: list[int] = []
        for offset, row in enumerate(chunk_rows):
            movie_id = start + offset
            for piece in plot_pieces(row.get("Plot") or ""):
                texts.append(f"{row.get('Title') or ''}. {piece}")
                local_owners.append(movie_id)
        vectors = model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings[cursor : cursor + len(texts)] = vectors.astype(np.float16)
        for vector, movie_id in zip(vectors, local_owners):
            centroids[movie_id] += vector
            counts[movie_id] += 1
        cursor += len(texts)
        embeddings.flush()
        print(f"encoded movies {start + len(chunk_rows)}/{len(rows)} sentences {cursor}", flush=True)

    counts = np.maximum(counts, 1)
    centroids /= counts[:, None]
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(centroids).astype(np.float32)
    np.save(INDEX_DIR / "pca_coords.npy", coords)
    np.savez(
        INDEX_DIR / "pca_model.npz",
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained=pca.explained_variance_ratio_.astype(np.float32),
    )
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "movies": len(rows),
                "sentences": int(len(owner_array)),
                "dim": dim,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("pca", pca.explained_variance_ratio_.tolist(), flush=True)
    print("index ready", flush=True)


if __name__ == "__main__":
    main()
