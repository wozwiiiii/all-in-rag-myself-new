"""
PowerRAG (RAGFlow) SDK Demo configuration.

This module follows the `code/` directory convention:
- Provide a small config object
- Load `.env` automatically (if present)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass(frozen=True)
class PowerRAGDemoConfig:
    base_url: str = os.getenv("RAGFLOW_BASE_URL", "http://127.0.0.1:9380").strip()
    api_key: str = os.getenv("RAGFLOW_API_KEY", "").strip()
    dataset_name: str = os.getenv("RAGFLOW_DATASET_NAME", "powerrag_text_qa_demo").strip()
    embedding_model: str = os.getenv("RAGFLOW_EMBEDDING_MODEL", "").strip()

    top_k: int = int(os.getenv("RAGFLOW_TOP_K", "5"))
    candidate_k: int = int(os.getenv("RAGFLOW_CANDIDATE_K", "1024"))
    similarity_threshold: float = float(os.getenv("RAGFLOW_SIMILARITY_THRESHOLD", "0.2"))
    vector_similarity_weight: float = float(os.getenv("RAGFLOW_VECTOR_SIMILARITY_WEIGHT", "0.3"))
    keyword: bool = _bool_env("RAGFLOW_KEYWORD", False)


DEFAULT_CONFIG = PowerRAGDemoConfig()

