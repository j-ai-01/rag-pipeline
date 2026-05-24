from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
INGEST_LOG = BASE_DIR / ".ingested_files.json"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
VISION_MODEL = "llava"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
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
