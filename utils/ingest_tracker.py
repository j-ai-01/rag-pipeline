import json
from pathlib import Path
from config import INGEST_LOG


def load_ingested() -> dict:
    if INGEST_LOG.exists():
        return json.loads(INGEST_LOG.read_text())
    return {}


def save_ingested(ingested: dict) -> None:
    INGEST_LOG.write_text(json.dumps(ingested, indent=2))


def is_already_ingested(file_hash: str, ingested: dict) -> bool:
    return file_hash in ingested.values()
