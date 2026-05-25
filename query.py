import sys
import io
import os
import logging
import warnings
import argparse

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

from typing import List, Optional

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
    EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL,
    TOP_K, HYBRID_ALPHA, INDEXES_DIR,
    get_index_paths,
)
from utils.ollama_check import assert_ollama_running
from utils.chroma_client import make_chroma_client
from utils.hybrid_retriever import HybridRetriever
from utils.multi_index_retriever import MultiIndexRetriever


def list_indexed() -> List[str]:
    if not INDEXES_DIR.exists():
        return []
    return sorted(
        d.name for d in INDEXES_DIR.iterdir()
        if d.is_dir() and (d / "docstore.json").exists()
    )


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
        index_name = meta.get("index_name", "")
        tag = f" [{index_name}]" if index_name else ""
        if page:
            key = f"{filename}:p{page}:{index_name}"
            label = f"  - {filename} (page {page}){tag}"
        elif file_type == "image":
            key = f"{filename}:img:{index_name}"
            label = f"  - {filename} (image description){tag}"
        else:
            key = f"{filename}:{index_name}"
            label = f"  - {filename}{tag}"
        if key not in seen:
            seen.add(key)
            lines.append(label)
    return "\n".join(lines) if lines else "No sources found."


def build_retriever(index_name: str) -> AutoMergingRetriever:
    paths = get_index_paths(index_name)
    if not paths.docstore_path.exists():
        raise FileNotFoundError(
            f"Index '{index_name}' not found. "
            f"Run `python ingest.py --index {index_name}` first."
        )

    docstore = SimpleDocumentStore.from_persist_path(str(paths.docstore_path))

    chroma_client = make_chroma_client(str(paths.chroma_dir))
    chroma_collection = chroma_client.get_or_create_collection(paths.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
    )

    top_k = min(TOP_K, chroma_collection.count()) or 1

    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    bm25_retriever = None
    if paths.bm25_index_path.exists():
        try:
            bm25_retriever = BM25Retriever.from_persist_dir(str(paths.bm25_index_path))
            bm25_retriever.similarity_top_k = top_k
        except Exception:
            print(f"Warning: [{index_name}] Could not load BM25 index. Using vector-only retrieval.")
    else:
        print(f"Warning: [{index_name}] BM25 index not found. Using vector-only retrieval.")

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        alpha=HYBRID_ALPHA,
        top_k=top_k,
    )

    return AutoMergingRetriever(
        hybrid_retriever,
        storage_context,
        simple_ratio_thresh=0.4,
        verbose=False,
    )


def build_query_engine(index_names: List[str]) -> RetrieverQueryEngine:
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120.0)

    loaded = []
    for name in index_names:
        try:
            loaded.append((name, build_retriever(name)))
        except FileNotFoundError as e:
            print(f"Warning: {e}  Skipping index '{name}'.")

    if not loaded:
        raise RuntimeError("No indexes could be loaded.")

    if len(loaded) == 1:
        retriever = loaded[0][1]
    else:
        retriever = MultiIndexRetriever(retrievers=loaded, top_k=TOP_K)

    return RetrieverQueryEngine.from_args(retriever=retriever, llm=Settings.llm)


def run_query(question: str, index_names: Optional[List[str]] = None) -> None:
    assert_ollama_running()

    if index_names is None:
        index_names = list_indexed()

    if not index_names:
        print("No indexes found. Create one with `python ingest.py --index <name>`.")
        return

    print(f"\nSearching for: {question}\n")
    try:
        engine = build_query_engine(index_names)
    except RuntimeError as e:
        print(str(e))
        return

    response = engine.query(question)
    answer = response.response or "Could not generate a summary. See sources below."
    sources = format_sources(response.source_nodes)

    print(f"Answer:\n{answer}\n")
    print(f"Sources:\n{sources}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query one or more named indexes.")
    parser.add_argument(
        "--index", default=None,
        help="Comma-separated index names to query (default: all indexed)",
    )
    parser.add_argument("question", nargs="+")
    args = parser.parse_args()
    names = [n.strip() for n in args.index.split(",")] if args.index else None
    run_query(" ".join(args.question), index_names=names)
