from unittest.mock import patch, MagicMock
from query import format_sources, run_query, list_indexed


def test_format_sources_with_pdf():
    node = MagicMock()
    node.metadata = {"filename": "report.pdf", "page_number": 3, "file_type": "pdf", "index_name": "finance"}
    result = format_sources([node])
    assert "report.pdf" in result
    assert "page 3" in result
    assert "[finance]" in result


def test_format_sources_with_text():
    node = MagicMock()
    node.metadata = {"filename": "notes.md", "file_type": "text", "index_name": "hr"}
    result = format_sources([node])
    assert "notes.md" in result
    assert "[hr]" in result


def test_format_sources_with_image():
    node = MagicMock()
    node.metadata = {"filename": "diagram.png", "file_type": "image", "index_name": "default"}
    result = format_sources([node])
    assert "diagram.png" in result
    assert "image" in result.lower()
    assert "[default]" in result


def test_format_sources_empty_returns_none_string():
    result = format_sources([])
    assert result == "No sources found."


def test_format_sources_no_index_name_omits_tag():
    node = MagicMock()
    node.metadata = {"filename": "notes.md", "file_type": "text"}
    result = format_sources([node])
    assert "notes.md" in result
    assert "[" not in result


def test_run_query_prints_answer_and_sources(capsys):
    mock_node = MagicMock()
    mock_node.metadata = {
        "filename": "doc.pdf", "page_number": 1,
        "file_type": "pdf", "index_name": "default",
    }
    mock_response = MagicMock()
    mock_response.response = "The answer is 42."
    mock_response.source_nodes = [mock_node]

    mock_engine = MagicMock()
    mock_engine.query.return_value = mock_response

    with patch("query.assert_ollama_running"), \
         patch("query.list_indexed", return_value=["default"]), \
         patch("query.build_query_engine", return_value=mock_engine):
        run_query("What is the answer?")

    captured = capsys.readouterr()
    assert "The answer is 42." in captured.out
    assert "doc.pdf" in captured.out


def test_run_query_prints_message_when_no_indexes(capsys):
    with patch("query.assert_ollama_running"), \
         patch("query.list_indexed", return_value=[]):
        run_query("anything")
    captured = capsys.readouterr()
    assert "No indexes found" in captured.out


def test_list_indexed_returns_empty_when_indexes_dir_missing(tmp_path):
    with patch("query.INDEXES_DIR", tmp_path / "nonexistent"):
        result = list_indexed()
    assert result == []


def test_list_indexed_returns_only_ingested_indexes(tmp_path):
    (tmp_path / "finance" / "docstore.json").parent.mkdir(parents=True)
    (tmp_path / "finance" / "docstore.json").touch()
    (tmp_path / "hr").mkdir()  # exists but no docstore.json
    with patch("query.INDEXES_DIR", tmp_path):
        result = list_indexed()
    assert result == ["finance"]
    assert "hr" not in result
