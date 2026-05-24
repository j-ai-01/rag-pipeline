import sys
import io
import os
import pickle
import shutil
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

from pathlib import Path
from typing import List, Tuple

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

from config import (
    DATA_DIR, CHROMA_DIR, EMBED_MODEL, LLM_MODEL,
    OLLAMA_BASE_URL, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE, CHUNK_OVERLAP,
    COLLECTION_NAME, ALL_SUPPORTED, SUPPORTED_EXTENSIONS,
    DOCSTORE_PATH, BM25_INDEX_PATH, LEAF_NODES_PATH, INGEST_LOG, TOP_K,
)
from utils.ollama_check import assert_ollama_running
from utils.file_hash import file_hash
from utils.ingest_tracker import load_ingested, save_ingested, is_already_ingested
from utils.image_describer import describe_image
from utils.chroma_client import make_chroma_client


def collect_files(data_dir: Path) -> List[Path]:
    found = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in ALL_SUPPORTED:
            found.append(path)
        elif path.name != ".gitkeep":
            print(f"Skipping unsupported file: {path.name}")
    return found


def build_documents(
    files: List[Path], ingested: dict
) -> Tuple[List[Document], dict]:
    documents = []
    updated_ingested = dict(ingested)

    for path in files:
        h = file_hash(path)
        if is_already_ingested(h, ingested):
            print(f"  [skip] {path.name} — already indexed")
            continue

        print(f"  [ingest] {path.name}")
        ext = path.suffix.lower()

        try:
            if ext in SUPPORTED_EXTENSIONS["image"]:
                text = describe_image(path)
                file_type = "image"
            elif ext in SUPPORTED_EXTENSIONS["pdf"]:
                from llama_index.readers.file import PDFReader
                reader = PDFReader()
                loaded = reader.load_data(file=path)
                for i, doc in enumerate(loaded):
                    doc.metadata.update({
                        "filename": path.name,
                        "file_type": "pdf",
                        "page_number": i + 1,
                    })
                documents.extend(loaded)
                updated_ingested[path.name] = h
                continue
            elif ext in SUPPORTED_EXTENSIONS["docx"]:
                from llama_index.readers.file import DocxReader
                reader = DocxReader()
                loaded = reader.load_data(file=path)
                for doc in loaded:
                    doc.metadata.update({"filename": path.name, "file_type": "docx"})
                documents.extend(loaded)
                updated_ingested[path.name] = h
                continue
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                file_type = "text"

            doc = Document(
                text=text,
                metadata={"filename": path.name, "file_type": file_type},
            )
            documents.append(doc)
            updated_ingested[path.name] = h

        except Exception as e:
            print(f"  [error] Could not process {path.name}: {e}")

    return documents, updated_ingested


def _wipe_index() -> None:
    for path in [CHROMA_DIR, DOCSTORE_PATH, BM25_INDEX_PATH, LEAF_NODES_PATH, INGEST_LOG]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    print("Index wiped. Re-indexing all files...")


def run_ingestion(reindex: bool = False) -> None:
    assert_ollama_running()

    if reindex:
        _wipe_index()

    DATA_DIR.mkdir(exist_ok=True)

    files = collect_files(DATA_DIR)
    if not files:
        print("No files found in data/. Add files and re-run ingest.py")
        return

    ingested = load_ingested()
    print(f"Found {len(files)} file(s). Checking for new files...")

    documents, updated_ingested = build_documents(files, ingested)

    if not documents:
        print("No new files to ingest. Everything is up to date.")
        return

    print(f"\nParsing {len(documents)} document(s) into hierarchical chunks...")

    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE],
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_nodes = parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)
    print(f"  {len(all_nodes)} total nodes, {len(leaf_nodes)} child (leaf) nodes")

    # Load or create docstore and add all new nodes
    if DOCSTORE_PATH.exists():
        docstore = SimpleDocumentStore.from_persist_path(str(DOCSTORE_PATH))
    else:
        docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)

    # Embed child nodes into ChromaDB
    CHROMA_DIR.mkdir(exist_ok=True)
    chroma_client = make_chroma_client(str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
    )

    print(f"Embedding {len(leaf_nodes)} child chunks via Ollama ({EMBED_MODEL})...")
    VectorStoreIndex(leaf_nodes, storage_context=storage_context, show_progress=True)

    # Persist docstore
    docstore.persist(str(DOCSTORE_PATH))

    # Accumulate all leaf nodes and rebuild BM25 index from all of them
    existing_leaf_nodes: list = []
    if LEAF_NODES_PATH.exists():
        with open(LEAF_NODES_PATH, "rb") as f:
            existing_leaf_nodes = pickle.load(f)
    all_leaf_nodes = existing_leaf_nodes + leaf_nodes
    with open(LEAF_NODES_PATH, "wb") as f:
        pickle.dump(all_leaf_nodes, f)

    print("Building BM25 keyword index...")
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=all_leaf_nodes,
        similarity_top_k=TOP_K,
    )
    bm25_retriever.persist(str(BM25_INDEX_PATH))

    save_ingested(updated_ingested)
    print(f"\nDone. {len(documents)} document(s) indexed ({len(leaf_nodes)} child chunks embedded).")


if __name__ == "__main__":
    reindex = "--reindex" in sys.argv
    run_ingestion(reindex=reindex)
