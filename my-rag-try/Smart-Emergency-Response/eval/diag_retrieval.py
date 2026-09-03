"""检索质量诊断：隔离检索与生成，直接测关键词召回率。

对 text 类题目逐题测三路排名（全量精确 cosine / BM25 / RRF 融合），
统计全关键词命中的首个 chunk 排名——判断失分来自检索还是生成。

    python eval/diag_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.rag.bm25 import BM25Index, rrf_fuse  # noqa: E402

TOP_K = 20


def main() -> None:
    setup_logging()
    settings = get_settings()
    llm = LLMClient(settings)

    import chromadb

    from app.rag.ingest import COLLECTION_NAME

    col = chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(COLLECTION_NAME)
    got = col.get(include=["documents"])
    docs = got["documents"]

    # 向量路：全量精确 cosine（对齐 retriever 实现）
    emb = np.asarray(col.get(include=["embeddings"])["embeddings"], dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    bm25 = BM25Index(docs)

    dataset = []
    with open(EVAL_DIR / "dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    print(f"{'id':>3} {'cat':<7} {'vec':>5} {'bm25':>5} {'rrf':>5}  keywords")
    for item in dataset:
        if item["category"] not in ("text", "hybrid"):
            continue
        q, kws = item["question"], item["keywords"]
        if not kws:
            continue

        qvec = np.asarray(llm.embed_query(q), dtype=np.float32)
        qvec = qvec / np.linalg.norm(qvec)
        sims = emb @ qvec
        vec_rank = list(np.argsort(-sims))  # 全量降序
        bm25_hits = bm25.search(q, top_k=len(docs))
        bm25_rank = [i for i, _ in bm25_hits]
        fused = rrf_fuse([vec_rank[:TOP_K], bm25_rank[:TOP_K]])

        def rank_all_kw(rank_list) -> int | None:
            for pos, i in enumerate(rank_list, start=1):
                if all(k in docs[i] for k in kws):
                    return pos
            return None

        r_vec = rank_all_kw(vec_rank[:TOP_K])
        r_bm25 = rank_all_kw(bm25_rank[:TOP_K]) if bm25_rank else None
        r_fused = rank_all_kw(fused)
        print(f"{item['id']:>3} {item['category']:<7} "
              f"{str(r_vec):>5} {str(r_bm25):>5} {str(r_fused):>5}  {','.join(kws)}")


if __name__ == "__main__":
    main()
