from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).parent
INDEXES_DIR = BASE_DIR / "indexes"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
VISION_MODEL = "llava"

PARENT_CHUNK_SIZE = 1024
CHILD_CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
TOP_K = 5
HYBRID_ALPHA = 0.5

SUPPORTED_EXTENSIONS = {
    "text": {".txt", ".md"},
    "pdf": {".pdf"},
    "docx": {".docx"},
    "image": {".jpg", ".jpeg", ".png", ".gif", ".webp"},
}

ALL_SUPPORTED = (
    SUPPORTED_EXTENSIONS["text"]
    | SUPPORTED_EXTENSIONS["pdf"]
    | SUPPORTED_EXTENSIONS["docx"]
    | SUPPORTED_EXTENSIONS["image"]
)


@dataclass
class IndexPaths:
    data_dir: Path
    chroma_dir: Path
    docstore_path: Path
    bm25_index_path: Path
    leaf_nodes_path: Path
    ingest_log: Path
    collection_name: str


def get_index_paths(name: str) -> IndexPaths:
    root = INDEXES_DIR / name
    return IndexPaths(
        data_dir=root / "data",
        chroma_dir=root / "chroma_db",
        docstore_path=root / "docstore.json",
        bm25_index_path=root / "bm25_index.pkl",
        leaf_nodes_path=root / "leaf_nodes.pkl",
        ingest_log=root / ".ingested_files.json",
        collection_name=f"rag_{name}",
    )
