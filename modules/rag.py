"""LlamaIndex RAG for Ask/AI — offline retrieval on user's own data."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def rag_available() -> bool:
    try:
        from llama_index.core import Document, VectorStoreIndex  # noqa: F401
        return True
    except ImportError:
        return False


def build_index(df: pd.DataFrame, domain: str = "generic") -> Any:
    """Build a LlamaIndex VectorStoreIndex from dataframe rows."""
    try:
        from llama_index.core import Document, VectorStoreIndex, Settings
        from llama_index.core.node_parser import SentenceSplitter

        docs = []
        for i, row in df.head(200).iterrows():
            text = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            docs.append(Document(text=text, metadata={"row": int(i), "domain": domain}))

        Settings.chunk_size = 256
        index = VectorStoreIndex.from_documents(docs)
        return index
    except ImportError:
        return None
    except Exception:
        return None


def query_rag(index: Any, question: str) -> dict[str, Any]:
    """Query the RAG index."""
    if index is None:
        return _keyword_fallback(question)
    try:
        engine = index.as_query_engine()
        response = engine.query(question)
        return {
            "answer": str(response),
            "source": "rag_llamaindex",
        }
    except Exception as e:
        return {"answer": f"RAG query failed: {e}", "source": "rag_error"}


def _keyword_fallback(question: str) -> dict[str, Any]:
    """Simple keyword-based fallback when LlamaIndex not available."""
    return {
        "answer": f"[RAG stub] Keyword search for: '{question}'. Install llama-index-core for full RAG.",
        "source": "rag_keyword_stub",
    }
