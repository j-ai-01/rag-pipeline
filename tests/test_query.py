from unittest.mock import patch, MagicMock
from query import format_sources, run_query


def test_format_sources_with_pdf():
    node = MagicMock()
    node.metadata = {"filename": "report.pdf", "page_number": 3, "file_type": "pdf"}
    result = format_sources([node])
    assert "report.pdf" in result
    assert "page 3" in result


def test_format_sources_with_text():
    node = MagicMock()
    node.metadata = {"filename": "notes.md", "file_type": "text"}
    result = format_sources([node])
    assert "notes.md" in result


def test_format_sources_with_image():
    node = MagicMock()
    node.metadata = {"filename": "diagram.png", "file_type": "image"}
    result = format_sources([node])
    assert "diagram.png" in result
    assert "image" in result.lower()


def test_format_sources_empty_returns_none_string():
    result = format_sources([])
    assert result == "No sources found."


def test_run_query_prints_answer_and_sources(capsys):
    mock_node = MagicMock()
    mock_node.metadata = {"filename": "doc.pdf", "page_number": 1, "file_type": "pdf"}
    mock_response = MagicMock()
    mock_response.response = "The answer is 42."
    mock_response.source_nodes = [mock_node]

    mock_engine = MagicMock()
    mock_engine.query.return_value = mock_response

    with patch("query.assert_ollama_running"), \
         patch("query.build_query_engine", return_value=mock_engine):
        run_query("What is the answer?")

    captured = capsys.readouterr()
    assert "The answer is 42." in captured.out
    assert "doc.pdf" in captured.out
