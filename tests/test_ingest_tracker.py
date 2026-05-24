import json
from pathlib import Path
from utils.ingest_tracker import load_ingested, save_ingested, is_already_ingested


def test_load_returns_empty_dict_when_no_log(tmp_path):
    log_path = tmp_path / ".ingested.json"
    result = load_ingested(log_path)
    assert result == {}


def test_save_and_load_roundtrip(tmp_path):
    log_path = tmp_path / ".ingested.json"
    save_ingested({"file.pdf": "abc123"}, log_path)
    result = load_ingested(log_path)
    assert result == {"file.pdf": "abc123"}


def test_is_already_ingested_true_when_hash_present():
    ingested = {"report.pdf": "deadbeef"}
    assert is_already_ingested("deadbeef", ingested) is True


def test_is_already_ingested_false_when_hash_missing():
    ingested = {"report.pdf": "deadbeef"}
    assert is_already_ingested("cafebabe", ingested) is False
