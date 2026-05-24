from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
INGEST_LOG = BASE_DIR / ".ingested_files.json"
DOCSTORE_PATH = BASE_DIR / "docstore.json"
BM25_INDEX_PATH = BASE_DIR / "bm25_index.pkl"
LEAF_NODES_PATH = BASE_DIR / "leaf_nodes.pkl"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
VISION_MODEL = "llava"

PARENT_CHUNK_SIZE = 1024
CHILD_CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
TOP_K = 5
HYBRID_ALPHA = 0.5    # 0.0 = pure BM25, 1.0 = pure vector
COLLECTION_NAME = "rag_docs"

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
