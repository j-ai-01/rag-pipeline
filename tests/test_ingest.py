from pathlib import Path
from unittest.mock import patch, MagicMock
from config import IndexPaths
from ingest import collect_files, build_documents, run_ingestion


def _fake_paths(tmp_path: Path) -> IndexPaths:
    return IndexPaths(
        data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma_db",
        docstore_path=tmp_path / "docstore.json",
        bm25_index_path=tmp_path / "bm25_index.pkl",
        leaf_nodes_path=tmp_path / "leaf_nodes.pkl",
        ingest_log=tmp_path / ".ingested_files.json",
        collection_name="rag_test",
    )


def test_collect_files_finds_supported_types(tmp_path):
    (tmp_path / "doc.pdf").touch()
    (tmp_path / "notes.md").touch()
    (tmp_path / "image.png").touch()
    (tmp_path / "unknown.xyz").touch()
    files = collect_files(tmp_path)
    names = {f.name for f in files}
    assert "doc.pdf" in names
    assert "notes.md" in names
    assert "image.png" in names
    assert "unknown.xyz" not in names


def test_collect_files_empty_dir_returns_empty(tmp_path):
    assert collect_files(tmp_path) == []


def test_collect_files_skips_unsupported_with_warning(tmp_path, capsys):
    (tmp_path / "weird.xyz").touch()
    collect_files(tmp_path)
    captured = capsys.readouterr()
    assert "weird.xyz" in captured.out or "Skipping" in captured.out


def test_build_documents_skips_already_ingested(tmp_path):
    f = tmp_path / "report.pdf"
    f.write_bytes(b"pdf content")
    from utils.file_hash import file_hash
    h = file_hash(f)
    ingested = {"report.pdf": h}
    docs, updated = build_documents([f], ingested)
    assert docs == []
    assert updated == ingested


def test_run_ingestion_prints_message_when_no_files(tmp_path, capsys):
    with patch("ingest.get_index_paths", return_value=_fake_paths(tmp_path)), \
         patch("ingest.assert_ollama_running"):
        run_ingestion()
    captured = capsys.readouterr()
    assert "No files found" in captured.out
