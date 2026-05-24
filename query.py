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
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_DIR, DOCSTORE_PATH, BM25_INDEX_PATH,
    EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL,
    TOP_K, COLLECTION_NAME, HYBRID_ALPHA,
)
from utils.ollama_check import assert_ollama_running
from utils.chroma_client import make_chroma_client
from utils.hybrid_retriever import HybridRetriever


def format_sources(source_nodes: List[NodeWithScore]) -> str:
    if not source_nodes:
        return "No sources found."
    lines = []
    seen = set()
    for node in source_nodes:
        meta = node.metadata
        filename = meta.get("filename", "unknown")
        file_type = meta.get("file_type", "")
        page = meta.get("page_number")
        if page:
            key = f"{filename}:p{page}"
            label = f"  - {filename} (page {page})"
        elif file_type == "image":
            key = f"{filename}:img"
            label = f"  - {filename} (image description)"
        else:
            key = filename
            label = f"  - {filename}"
        if key not in seen:
            seen.add(key)
            lines.append(label)
    return "\n".join(lines) if lines else "No sources found."


def build_query_engine():
    if not DOCSTORE_PATH.exists():
        raise FileNotFoundError(
            "No index found. Run `python ingest.py` first to index your documents."
        )

    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120.0)

    # Load docstore (parent + child nodes)
    docstore = SimpleDocumentStore.from_persist_path(str(DOCSTORE_PATH))

    # Set up ChromaDB vector store
    chroma_client = make_chroma_client(str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
    )

    top_k = min(TOP_K, chroma_collection.count()) or 1

    # Vector retriever (child nodes from ChromaDB)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    # BM25 retriever (optional — falls back to vector-only if missing)
    bm25_retriever = None
    if BM25_INDEX_PATH.exists():
        try:
            bm25_retriever = BM25Retriever.from_persist_dir(str(BM25_INDEX_PATH))
            bm25_retriever.similarity_top_k = top_k
        except Exception:
            print("Warning: Could not load BM25 index. Using vector-only retrieval.")
    else:
        print("Warning: BM25 index not found. Using vector-only retrieval.")

    # Hybrid retriever: fuses vector + BM25 child results
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        alpha=HYBRID_ALPHA,
        top_k=top_k,
    )

    # Auto-merge: expands matched child nodes to their parent nodes
    auto_merging_retriever = AutoMergingRetriever(
        hybrid_retriever,
        storage_context,
        simple_ratio_thresh=0.4,
        verbose=False,
    )

    return RetrieverQueryEngine.from_args(
        retriever=auto_merging_retriever,
        llm=Settings.llm,
    )


def run_query(question: str) -> None:
    assert_ollama_running()

    if not CHROMA_DIR.exists() and not DOCSTORE_PATH.exists():
        print("No index found. Run ingest.py first to index your documents.")
        return

    print(f"\nSearching for: {question}\n")
    try:
        engine = build_query_engine()
    except FileNotFoundError as e:
        print(str(e))
        return

    response = engine.query(question)
    answer = response.response or "Could not generate a summary. See sources below."
    sources = format_sources(response.source_nodes)

    print(f"Answer:\n{answer}\n")
    print(f"Sources:\n{sources}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)
    run_query(" ".join(sys.argv[1:]))
