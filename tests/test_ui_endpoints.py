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
    _meta = {
        "filename": filename,
        "page_number": page,
        "file_type": "pdf",
        "index_name": index,
    }
    mock_node.metadata = _meta
    mock_node.node.metadata = _meta
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


def test_get_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RAG Search" in response.text


def test_post_query_returns_snippets():
    mock_engine = MagicMock()
    mock_engine.query.return_value = _make_mock_response()

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=["docs"]), \
         patch("mcp_server.build_query_engine", return_value=mock_engine):
        response = client.post("/query", json={"question": "What is 42?"})

    assert response.status_code == 200
    data = response.json()
    assert "snippets" in data
    assert isinstance(data["snippets"], list)
    assert len(data["snippets"]) == 1
    assert data["snippets"][0]["filename"] == "doc.pdf"
    assert "text" in data["snippets"][0]


def test_post_query_stream_returns_event_sequence():
    mock_node = MagicMock()
    mock_node.node.metadata = {"filename": "doc.pdf", "page_number": 1, "index_name": "docs"}
    mock_node.node.get_content.return_value = "Full chunk text for snippet"
    mock_node.score = 0.9

    mock_streaming_resp = MagicMock()
    mock_streaming_resp.response_gen = iter(["The ", "answer ", "is 42."])

    mock_synthesizer = MagicMock()
    mock_synthesizer.synthesize.return_value = mock_streaming_resp

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=["docs"]), \
         patch("mcp_server.build_retriever") as mock_br, \
         patch("mcp_server.get_response_synthesizer", return_value=mock_synthesizer):
        mock_br.return_value.retrieve.return_value = [mock_node]
        response = client.post("/query/stream", json={"question": "What is 42?"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    import json as _json
    events = [
        _json.loads(line[6:])
        for line in response.text.split("\n\n")
        if line.startswith("data: ")
    ]

    assert events[0]["type"] == "sources"
    assert events[0]["snippets"][0]["filename"] == "doc.pdf"
    assert events[0]["snippets"][0]["text"] == "Full chunk text for snippet"
    token_texts = [e["text"] for e in events if e["type"] == "token"]
    assert token_texts == ["The ", "answer ", "is 42."]
    assert events[-1]["type"] == "done"


def test_post_query_stream_error_when_ollama_not_running():
    import json as _json

    with patch("mcp_server.check_ollama_running", return_value=False):
        response = client.post("/query/stream", json={"question": "hi"})

    events = [
        _json.loads(line[6:])
        for line in response.text.split("\n\n")
        if line.startswith("data: ")
    ]
    assert events[0]["type"] == "error"
    assert "Ollama" in events[0]["message"]


def test_post_query_stream_error_when_no_indexes():
    import json as _json

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=[]):
        response = client.post("/query/stream", json={"question": "hi"})

    events = [
        _json.loads(line[6:])
        for line in response.text.split("\n\n")
        if line.startswith("data: ")
    ]
    assert events[0]["type"] == "error"
    assert "No indexes" in events[0]["message"]
