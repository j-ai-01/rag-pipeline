import sys
from pathlib import Path
from typing import List, Tuple

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    DATA_DIR, CHROMA_DIR, EMBED_MODEL, LLM_MODEL,
    OLLAMA_BASE_URL, CHUNK_SIZE, CHUNK_OVERLAP,
    COLLECTION_NAME, ALL_SUPPORTED, SUPPORTED_EXTENSIONS,
)
from utils.ollama_check import assert_ollama_running
from utils.file_hash import file_hash
from utils.ingest_tracker import load_ingested, save_ingested, is_already_ingested
from utils.image_describer import describe_image


def collect_files(data_dir: Path) -> List[Path]:
    """Scan data_dir and return supported files. Warn about unsupported."""
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
    """Parse files into LlamaIndex Documents. Skip already-ingested files."""
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


def run_ingestion() -> None:
    assert_ollama_running()

    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)

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

    print(f"\nEmbedding {len(documents)} document(s) via Ollama ({EMBED_MODEL})...")

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL
    )
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    save_ingested(updated_ingested)
    print(f"\nDone. {len(documents)} document(s) indexed into ChromaDB.")


if __name__ == "__main__":
    run_ingestion()
