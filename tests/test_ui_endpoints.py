from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from mcp_server import app

client = TestClient(app)


def test_get_indexes_returns_list():
    with patch("mcp_server.list_indexed", return_value=["docs", "reports"]):
        response = client.get("/indexes")
    assert response.status_code == 200
    assert response.json() == ["docs", "reports"]


def test_get_indexes_returns_empty_list():
    with patch("mcp_server.list_indexed", return_value=[]):
        response = client.get("/indexes")
    assert response.status_code == 200
    assert response.json() == []


def _make_mock_response(answer="The answer is 42.", filename="doc.pdf", page=1, index="docs"):
    mock_node = MagicMock()
    mock_node.metadata = {
        "filename": filename,
        "page_number": page,
        "file_type": "pdf",
        "index_name": index,
    }
    mock_response = MagicMock()
    mock_response.response = answer
    mock_response.source_nodes = [mock_node]
    return mock_response


def test_post_query_returns_answer_and_sources():
    mock_engine = MagicMock()
    mock_engine.query.return_value = _make_mock_response()

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=["docs"]), \
         patch("mcp_server.build_query_engine", return_value=mock_engine):
        response = client.post("/query", json={"question": "What is 42?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The answer is 42."
    assert "doc.pdf" in data["sources"]


def test_post_query_passes_selected_indexes():
    mock_engine = MagicMock()
    mock_engine.query.return_value = _make_mock_response()

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.build_query_engine", return_value=mock_engine) as mock_build:
        response = client.post("/query", json={"question": "hi", "indexes": ["reports"]})

    mock_build.assert_called_once_with(["reports"])
    assert response.status_code == 200
    assert "answer" in response.json()


def test_post_query_uses_all_indexes_when_none_selected():
    mock_engine = MagicMock()
    mock_engine.query.return_value = _make_mock_response()

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=["docs", "reports"]), \
         patch("mcp_server.build_query_engine", return_value=mock_engine) as mock_build:
        client.post("/query", json={"question": "hi"})

    mock_build.assert_called_once_with(["docs", "reports"])


def test_post_query_returns_error_when_ollama_not_running():
    with patch("mcp_server.check_ollama_running", return_value=False):
        response = client.post("/query", json={"question": "hi"})

    assert response.status_code == 503
    assert "error" in response.json()
    assert "Ollama" in response.json()["error"]


def test_post_query_returns_error_when_no_indexes():
    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=[]):
        response = client.post("/query", json={"question": "hi"})

    assert response.status_code == 404
    assert "error" in response.json()
    assert "No indexes" in response.json()["error"]


def test_post_query_rejects_empty_question():
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_post_query_returns_error_on_engine_exception():
    mock_engine = MagicMock()
    mock_engine.query.side_effect = RuntimeError("LLM timeout")

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=["docs"]), \
         patch("mcp_server.build_query_engine", return_value=mock_engine):
        response = client.post("/query", json={"question": "hi"})

    assert response.status_code == 500
    assert "error" in response.json()
    assert "LLM timeout" in response.json()["error"]
