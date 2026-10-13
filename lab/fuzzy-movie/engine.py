"""电影模糊检索。

每部电影的剧情被切成短句，分别做成句向量。
查询若有多个分句，就让每个分句各自找最像的一句剧情，再把分数加起来。
这样“做梦”和“陀螺”必须同时对上，而不是只碰到其中一个词。
BM25 仍按原始问句的字面来做，用来和语义检索对照。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from textutil import matched_phrases, plot_pieces, query_clauses

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"
DATA_CANDIDATES = [
    ROOT / "data" / "wiki_movie_plots_deduped.csv",
    ROOT / "data" / "wiki_movie_plots_deduped_with_summaries.csv",
]
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PHRASE_BONUS = 0.35


def find_csv() -> Path | None:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def tokenize(text: str) -> list[str]:
    import re

    lowered = (text or "").lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    characters = re.findall(r"[\u4e00-\u9fff]", text or "")
    # 单个汉字在英文剧情里偶尔出现，IDF 会虚高。中文按二字词匹配。
    bigrams = [characters[i] + characters[i + 1] for i in range(len(characters) - 1)]
    return words + bigrams


class MovieSearch:
    def __init__(self) -> None:
        meta_path = INDEX_DIR / "movies.jsonl"
        emb_path = INDEX_DIR / "sentence_emb.f16.npy"
        owner_path = INDEX_DIR / "sentence_owner.npy"
        if not meta_path.exists() or not emb_path.exists():
            raise FileNotFoundError("索引不存在。请先运行 python build_index.py")

        self.movies: list[dict] = []
        with meta_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.movies.append(json.loads(line))

        self.embeddings = np.asarray(np.load(emb_path), dtype=np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings /= np.maximum(norms, 1e-8)
        self.owners = np.asarray(np.load(owner_path), dtype=np.int32)

        self.coords = np.load(INDEX_DIR / "pca_coords.npy")
        pca = np.load(INDEX_DIR / "pca_model.npz")
        self.pca_mean = pca["mean"]
        self.pca_components = pca["components"]
        self.pca_explained = pca["explained"]

        self.plots = self._load_plots()
        corpus = [
            tokenize(f"{movie['title']} {movie['genre']} {self.plots[movie['id']]}")
            for movie in self.movies
        ]
        self.bm25 = BM25Okapi(corpus)
        self._model = None

    def _load_plots(self) -> list[str]:
        plots = [movie.get("plot_preview", "") for movie in self.movies]
        csv_path = find_csv()
        if csv_path is None:
            self.plot_source = "preview"
            return plots
        csv.field_size_limit(10**7)
        loaded: list[str] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                loaded.append(row.get("Plot") or "")
        if len(loaded) != len(self.movies):
            self.plot_source = "preview"
            return plots
        self.plot_source = "plot"
        return loaded

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _max_over_movies(self, vector: np.ndarray) -> np.ndarray:
        sentence_scores = self.embeddings @ vector
        best = np.full(len(self.movies), -1.0, dtype=np.float32)
        np.maximum.at(best, self.owners, sentence_scores)
        return best

    def project(self, vector: np.ndarray) -> tuple[float, float]:
        xy = self.pca_components @ (vector - self.pca_mean)
        return float(xy[0]), float(xy[1])

    def _hit(self, index: int, score: float, source: str, extra: dict | None = None) -> dict:
        movie = self.movies[index]
        plot = self.plots[index] or movie.get("plot_preview", "")
        hit = {
            "id": movie["id"],
            "title": movie["title"],
            "year": movie["year"],
            "origin": movie["origin"],
            "director": movie["director"],
            "cast": movie["cast"],
            "genre": movie["genre"],
            "wiki": movie["wiki"],
            "plot": plot,
            "score": round(float(score), 4),
            "x": round(float(self.coords[index, 0]), 4),
            "y": round(float(self.coords[index, 1]), 4),
            "source": source,
        }
        if extra:
            hit.update(extra)
        return hit

    def search_dense(self, query: str, top_k: int) -> tuple[list[dict], np.ndarray]:
        parts, extras = query_clauses(query)
        encoded = self.encode(parts + extras)
        total = np.zeros(len(self.movies), dtype=np.float32)
        best_cosine = np.full(len(self.movies), -1.0, dtype=np.float32)
        for vector in encoded[: len(parts)]:
            local = self._max_over_movies(vector)
            total += local
            best_cosine = np.maximum(best_cosine, local)
        if extras:
            extra_best = np.full(len(self.movies), -1.0, dtype=np.float32)
            for vector in encoded[len(parts) :]:
                extra_best = np.maximum(extra_best, self._max_over_movies(vector))
            total += extra_best
            best_cosine = np.maximum(best_cosine, extra_best)

        phrases_by_movie: dict[int, list[str]] = {}
        for index, plot in enumerate(self.plots):
            found = matched_phrases(query, plot)
            if found:
                total[index] += PHRASE_BONUS * len(found)
                phrases_by_movie[index] = found

        candidate_count = min(len(total), max(top_k * 3, top_k + 8))
        if candidate_count >= len(total):
            order = np.argsort(-total)
        else:
            pick = np.argpartition(-total, candidate_count)[:candidate_count]
            order = pick[np.argsort(-total[pick])]
        hits = []
        seen: set[tuple[str, str]] = set()
        for index in order:
            movie = self.movies[int(index)]
            key = (movie["title"].strip().lower(), movie["year"])
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                self._hit(
                    int(index),
                    float(total[index]),
                    "dense",
                    {
                        "cosine": round(float(best_cosine[index]), 4),
                        "phrases": phrases_by_movie.get(int(index), []),
                    },
                )
            )
            if len(hits) >= top_k:
                break
        self._attach_evidence(query, parts + extras, hits)
        return hits, total

    def _attach_evidence(self, query: str, clauses: list[str], hits: list[dict]) -> None:
        if not hits:
            return
        clause_vectors = self.encode(clauses)
        for hit in hits:
            pieces = plot_pieces(self.plots[hit["id"]])
            if not pieces:
                hit["evidence"] = ""
                continue
            phrase_hit = next(
                (piece for piece in pieces if any(phrase in piece.lower() for phrase in hit.get("phrases") or [])),
                "",
            )
            if phrase_hit:
                hit["evidence"] = phrase_hit
                hit["evidence_cosine"] = None
                continue
            piece_vectors = self.encode([f"{hit['title']}. {piece}" for piece in pieces])
            sims = piece_vectors @ clause_vectors.T
            flat = int(np.argmax(sims))
            sentence_index, clause_index = divmod(flat, sims.shape[1])
            hit["evidence"] = pieces[sentence_index]
            hit["evidence_cosine"] = round(float(sims[sentence_index, clause_index]), 4)

    def search_bm25(self, query: str, top_k: int) -> list[dict]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            pick = np.argpartition(-scores, top_k)[:top_k]
            order = pick[np.argsort(-scores[pick])]
        hits = []
        for index in order[:top_k]:
            if scores[index] <= 0:
                break
            hits.append(self._hit(int(index), float(scores[index]), "bm25"))
        return hits

    def search(self, query: str, top_k: int = 8) -> dict:
        query = query.strip()
        dense, _scores = self.search_dense(query, top_k)
        bm25 = self.search_bm25(query, top_k)
        query_vector = self.encode([query])[0]
        x, y = self.project(query_vector)
        margin = 0.0
        if len(dense) >= 2:
            margin = dense[0]["score"] - dense[1]["score"]
        top_score = dense[0]["score"] if dense else 0.0
        return {
            "query": query,
            "dense": dense,
            "bm25": bm25,
            "query_point": {"x": round(x, 4), "y": round(y, 4)},
            "plot_source": self.plot_source,
            "model": MODEL_NAME,
            "margin": round(float(margin), 4),
            "confident": bool(top_score >= 1.05 and margin >= 0.02),
        }

    def universe(self, limit: int = 2800) -> dict:
        decades: dict[str, list[int]] = {}
        for movie in self.movies:
            year = movie.get("year") or ""
            decade = f"{year[:3]}0s" if len(year) >= 3 and year[:3].isdigit() else "未知"
            decades.setdefault(decade, []).append(movie["id"])
        rng = np.random.default_rng(7)
        quota = max(1, limit // max(len(decades), 1))
        chosen: list[int] = []
        for ids in decades.values():
            take = min(len(ids), quota)
            picked = rng.choice(ids, size=take, replace=False)
            chosen.extend(int(item) for item in np.atleast_1d(picked))
        if len(chosen) > limit:
            chosen = [int(item) for item in rng.choice(chosen, size=limit, replace=False)]
        points = []
        for index in chosen:
            movie = self.movies[index]
            points.append(
                {
                    "id": movie["id"],
                    "title": movie["title"],
                    "year": movie["year"],
                    "origin": movie["origin"],
                    "x": round(float(self.coords[index, 0]), 3),
                    "y": round(float(self.coords[index, 1]), 3),
                }
            )
        return {
            "points": points,
            "explained": [round(float(value), 4) for value in self.pca_explained],
            "count": len(self.movies),
        }
