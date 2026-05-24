import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from utils.ingest_tracker import load_ingested, save_ingested, is_already_ingested


def test_load_returns_empty_dict_when_no_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / ".ingested.json"
        with patch("utils.ingest_tracker.INGEST_LOG", log_path):
            result = load_ingested()
    assert result == {}


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / ".ingested.json"
        with patch("utils.ingest_tracker.INGEST_LOG", log_path):
            save_ingested({"file.pdf": "abc123"})
            result = load_ingested()
    assert result == {"file.pdf": "abc123"}


def test_is_already_ingested_true_when_hash_present():
    ingested = {"report.pdf": "deadbeef"}
    assert is_already_ingested("deadbeef", ingested) is True


def test_is_already_ingested_false_when_hash_missing():
    ingested = {"report.pdf": "deadbeef"}
    assert is_already_ingested("cafebabe", ingested) is False
