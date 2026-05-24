import sys
import io
import os
import logging
import warnings

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)


class _StderrFilter(io.TextIOWrapper):
    _blocked = ("Failed to send telemetry event",)

    def write(self, msg):
        if not any(s in msg for s in self._blocked):
            sys.__stderr__.write(msg)
        return len(msg)

    def flush(self):
        sys.__stderr__.flush()


sys.stderr = _StderrFilter(io.BytesIO())
from typing import List

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_DIR, EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, COLLECTION_NAME,
)
from utils.ollama_check import assert_ollama_running
from utils.chroma_client import make_chroma_client


def format_sources(source_nodes: List[NodeWithScore]) -> str:
    if not source_nodes:
        return "No sources found."
    lines = []
    for node in source_nodes:
        meta = node.metadata
        filename = meta.get("filename", "unknown")
        file_type = meta.get("file_type", "")
        page = meta.get("page_number")
        if page:
            lines.append(f"  - {filename} (page {page})")
        elif file_type == "image":
            lines.append(f"  - {filename} (image description)")
        else:
            lines.append(f"  - {filename}")
    return "\n".join(lines)


def build_query_engine():
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL
    )
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    chroma_client = make_chroma_client(str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    top_k = min(TOP_K, chroma_collection.count()) or 1
    return index.as_query_engine(similarity_top_k=top_k)


def run_query(question: str) -> None:
    assert_ollama_running()

    if not CHROMA_DIR.exists():
        print("No index found. Run ingest.py first to index your documents.")
        return

    print(f"\nSearching for: {question}\n")
    engine = build_query_engine()
    response = engine.query(question)

    answer = response.response or "Could not generate a summary. See sources below."
    sources = format_sources(response.source_nodes)

    print(f"Answer:\n{answer}\n")
    print(f"Sources:\n{sources}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question here\"")
        sys.exit(1)
    run_query(" ".join(sys.argv[1:]))
