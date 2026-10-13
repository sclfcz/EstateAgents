"""模糊记忆：用情节描述找回电影。"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import MODEL_NAME, MovieSearch

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
POSTER_CACHE = ROOT / "index" / "poster_cache.json"

app = FastAPI(title="模糊记忆")
searcher: MovieSearch | None = None
poster_cache: dict[str, str] = {}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=400)
    top_k: int = Field(default=8, ge=1, le=20)


def get_searcher() -> MovieSearch:
    global searcher
    if searcher is None:
        searcher = MovieSearch()
    return searcher


@app.on_event("startup")
def startup() -> None:
    global poster_cache
    if POSTER_CACHE.exists():
        poster_cache = json.loads(POSTER_CACHE.read_text(encoding="utf-8"))
    get_searcher()


@app.get("/api/health")
def health() -> dict:
    engine = get_searcher()
    return {
        "movies": len(engine.movies),
        "model": MODEL_NAME,
        "plot_source": engine.plot_source,
    }


@app.get("/api/stats")
def stats() -> dict:
    engine = get_searcher()
    origins: dict[str, int] = {}
    decades: dict[str, int] = {}
    for movie in engine.movies:
        origins[movie["origin"] or "未知"] = origins.get(movie["origin"] or "未知", 0) + 1
        year = movie["year"]
        decade = f"{year[:3]}0s" if len(year) >= 3 and year[:3].isdigit() else "未知"
        decades[decade] = decades.get(decade, 0) + 1
    return {
        "movies": len(engine.movies),
        "origins": origins,
        "decades": decades,
        "plot_source": engine.plot_source,
        "pca_explained": [float(v) for v in engine.pca_explained],
    }


@app.post("/api/search")
def search(body: SearchRequest) -> dict:
    query = body.query.strip()
    if len(query) < 2:
        raise HTTPException(status_code=400, detail="请再多写一点情节，至少两个字。")
    return get_searcher().search(query, body.top_k)


@app.get("/api/universe")
def universe() -> dict:
    return get_searcher().universe()


@app.get("/api/poster")
def poster(wiki: str = Query(default="")) -> dict:
    wiki = wiki.strip()
    if not wiki:
        return {"image": ""}
    if wiki in poster_cache:
        return {"image": poster_cache[wiki]}
    title = urllib.parse.unquote(wiki.rstrip("/").split("/")[-1])
    api = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
    image = ""
    try:
        request = urllib.request.Request(api, headers={"User-Agent": "FuzzyMovieMemory/1.0 (student lab)"})
        with urllib.request.urlopen(request, timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))
        thumbnail = payload.get("thumbnail") or {}
        image = thumbnail.get("source") or ""
    except Exception:
        image = ""
    poster_cache[wiki] = image
    POSTER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    POSTER_CACHE.write_text(
        json.dumps(poster_cache, ensure_ascii=False), encoding="utf-8"
    )
    return {"image": image}


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
