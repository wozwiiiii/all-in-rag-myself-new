#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from config import DEFAULT_CONFIG


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _require(value: str | None, hint: str) -> str:
    if value is None or value.strip() == "":
        raise SystemExit(hint)
    return value.strip()


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise SystemExit(f"File not found: {path}")


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="PowerRAG (RAGFlow) SDK demo: upload Markdown, parse, retrieve top-k chunks.",
    )
    parser.add_argument("--file", type=Path, required=True, help="Markdown file path, e.g. ./data/sample.md")
    parser.add_argument("--question", type=str, required=True, help="User question for retrieval")
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k, help="How many chunks to return (mapped to page_size)")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_CONFIG.embedding_model or _env("RAGFLOW_EMBEDDING_MODEL"),
        help=(
            "Embedding model string in '<model>@<factory>' format. "
            "If omitted, server tenant default is used."
        ),
    )
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CONFIG.candidate_k, help="RAGFlow.retrieve(top_k=...) candidate pool size")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_CONFIG.similarity_threshold, help="Filter chunks below this similarity")
    parser.add_argument("--vector-similarity-weight", type=float, default=DEFAULT_CONFIG.vector_similarity_weight, help="Weight of vector similarity in hybrid score")
    parser.add_argument("--keyword", action="store_true", default=DEFAULT_CONFIG.keyword, help="Enable keyword matching (hybrid retrieval)")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_CONFIG.dataset_name, help="Dataset name to create")
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_CONFIG.base_url or _env("RAGFLOW_BASE_URL") or _env("POWERRAG_BASE_URL") or _env("BASE_URL"),
        help="RAGFlow/PowerRAG base_url (or env RAGFLOW_BASE_URL / POWERRAG_BASE_URL / BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_CONFIG.api_key or _env("RAGFLOW_API_KEY") or _env("POWERRAG_API_KEY") or _env("API_KEY"),
        help="RAGFlow/PowerRAG api_key (or env RAGFLOW_API_KEY / POWERRAG_API_KEY / API_KEY)",
    )
    parser.add_argument("--cleanup", action="store_true", help="Delete created dataset after finishing")

    args = parser.parse_args(argv)

    base_url = _require(args.base_url, "Missing base_url. Use --base-url or set env RAGFLOW_BASE_URL.")
    api_key = _require(args.api_key, "Missing api_key. Use --api-key or set env RAGFLOW_API_KEY.")

    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.candidate_k <= 0:
        raise SystemExit("--candidate-k must be > 0")

    blob = _read_bytes(args.file)
    display_name = args.file.name
    if not display_name.lower().endswith(".md"):
        display_name = f"{display_name}.md"

    try:
        from ragflow_sdk import RAGFlow  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Failed to import ragflow_sdk. Install dependencies first:\n"
            "  pip install -r requirements.txt\n"
            f"Original error: {e}"
        )

    rag = RAGFlow(api_key=api_key, base_url=base_url)

    dataset_kwargs: dict[str, Any] = {"name": args.dataset_name}
    if args.embedding_model:
        dataset_kwargs["embedding_model"] = args.embedding_model
    dataset = rag.create_dataset(**dataset_kwargs)
    try:
        docs = dataset.upload_documents([{"display_name": display_name, "blob": blob}])
        if not docs:
            raise SystemExit("Upload succeeded but no document returned by SDK.")
        doc = docs[0]

        parse_results = dataset.parse_documents([doc.id])
        # parse_results: list[tuple[doc_id, status, success_count, failure_count]] (per API ref)
        print("Parse results:")
        print(parse_results)
        if parse_results and isinstance(parse_results, list):
            statuses = {r[1] for r in parse_results if isinstance(r, (list, tuple)) and len(r) >= 2}
            if statuses and statuses != {"DONE"}:
                raise SystemExit(
                    "Document parsing failed (status not DONE). "
                    "Most common cause is missing/unauthorized embedding model.\n"
                    "Try:\n"
                    "  - set tenant default embedding model in UI or via /v1/user/set_tenant_info, OR\n"
                    "  - rerun with --embedding-model '<model>@<factory>' (must be supported & configured for the tenant)\n"
                    "If it still fails, check PowerRAG logs inside the container (task executor) for the detailed error.\n"
                )

        chunks = rag.retrieve(
            question=args.question,
            dataset_ids=[dataset.id],
            document_ids=[doc.id],
            page=1,
            page_size=args.top_k,
            similarity_threshold=args.similarity_threshold,
            vector_similarity_weight=args.vector_similarity_weight,
            top_k=args.candidate_k,
            keyword=args.keyword,
        )

        print("\nRetrieved chunks:")
        if not chunks:
            print("(empty)")
            return 0

        for i, c in enumerate(chunks, start=1):
            similarity = _safe_get(c, "similarity")
            vector_similarity = _safe_get(c, "vector_similarity")
            term_similarity = _safe_get(c, "term_similarity")
            content = _safe_get(c, "content", "")
            content_preview = (content or "").strip().replace("\n", " ")
            if len(content_preview) > 260:
                content_preview = content_preview[:260] + "…"
            print(f"{i:02d}. similarity={similarity} vector={vector_similarity} term={term_similarity}")
            print(f"    {content_preview}")

        return 0
    finally:
        if args.cleanup:
            try:
                rag.delete_datasets(ids=[dataset.id])
            except Exception as e:
                print(f"Warning: failed to cleanup dataset {dataset.id}: {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
