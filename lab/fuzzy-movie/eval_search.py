"""对若干手工查询比较稠密检索和 BM25 的名次。"""

from __future__ import annotations

import json
from pathlib import Path

from engine import MovieSearch

CASES = [
    {
        "query": "一个男人潜入别人的梦里偷取想法，最后陀螺一直在转",
        "expect": "Inception",
        "year": "2010",
    },
    {
        "query": "核战之后的废土上，有人开着改装卡车抢水",
        "expect": "Mad Max: Fury Road",
        "year": "2015",
    },
    {
        "query": "穷画家和富家女在撞上冰山的巨轮上相爱",
        "expect": "Titanic",
        "year": "1997",
    },
    {
        "query": "小女孩的父母变成了猪，她走进神灵世界想把他们变回来",
        "expect": "Spirited Away",
        "year": "2001",
    },
    {
        "query": "a thief enters dreams to plant an idea and checks a spinning top",
        "expect": "Inception",
        "year": "2010",
    },
    {
        "query": "post-apocalyptic desert convoy fighting for water",
        "expect": "Mad Max: Fury Road",
        "year": "2015",
    },
    {
        "query": "奥本海默组织科学家制造原子弹",
        "expect": None,
        "year": None,
    },
]


def rank_of(hits: list[dict], title: str, year: str | None) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if hit["title"] == title and (year is None or hit["year"] == year):
            return index
    return None


def main() -> None:
    engine = MovieSearch()
    rows = []
    for case in CASES:
        result = engine.search(case["query"], top_k=10)
        dense_rank = None
        bm25_rank = None
        if case["expect"]:
            dense_rank = rank_of(result["dense"], case["expect"], case["year"])
            bm25_rank = rank_of(result["bm25"], case["expect"], case["year"])
        rows.append(
            {
                "query": case["query"],
                "expect": case["expect"],
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "dense_top": [
                    f"{hit['title']} ({hit['year']}) {hit['score']}" for hit in result["dense"][:5]
                ],
                "bm25_top": [
                    f"{hit['title']} ({hit['year']}) {hit['score']}" for hit in result["bm25"][:5]
                ],
            }
        )
        print("=" * 20)
        print(case["query"])
        print("dense", rows[-1]["dense_top"])
        print("bm25", rows[-1]["bm25_top"])
    out = Path("report/eval_results.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
