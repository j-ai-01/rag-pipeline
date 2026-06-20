import json
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


def _make_mock_agent_response(answer="42 is the answer."):
    mock_node = MagicMock()
    mock_node.node.metadata = {"filename": "doc.pdf", "page_number": 1, "index_name": "docs"}
    mock_node.node.get_content.return_value = "This is the content of doc.pdf page 1."
    mock_node.score = 0.9

    mock_response = MagicMock()
    mock_response.response = answer
    mock_response.source_nodes = [mock_node]
    return mock_response


def _make_mock_streaming_response(tokens=("The ", "answer ", "is 42.")):
    mock_node = MagicMock()
    mock_node.node.metadata = {"filename": "doc.pdf", "page_number": 1, "index_name": "docs"}
    mock_node.node.get_content.return_value = "Full chunk text for snippet"
    mock_node.score = 0.9

    mock_streaming = MagicMock()
    mock_streaming.response_gen = iter(tokens)
    mock_streaming.source_nodes = [mock_node]
    return mock_streaming


def test_post_query_returns_answer_and_sources():
    mock_agent = MagicMock()
    mock_agent.chat.return_value = _make_mock_agent_response()

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=["docs"]), \
         patch("mcp_server.build_agent", return_value=mock_agent):
        response = client.post("/query", json={"question": "What is 42?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "42 is the answer."
    assert "sources" in data


def test_post_query_passes_selected_indexes():
    mock_agent = MagicMock()
    mock_agent.chat.return_value = _make_mock_agent_response()

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.build_agent", return_value=mock_agent) as mock_build:
        response = client.post("/query", json={"question": "What is 42?", "indexes": ["docs"]})

    assert response.status_code == 200
    mock_build.assert_called_once_with(["docs"])


def test_post_query_uses_all_indexes_when_none_selected():
    mock_agent = MagicMock()
    mock_agent.chat.return_value = _make_mock_agent_response()

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=["docs", "reports"]), \
         patch("mcp_server.build_agent", return_value=mock_agent) as mock_build:
        response = client.post("/query", json={"question": "What is 42?"})

    assert response.status_code == 200
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
    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=["docs"]), \
         patch("mcp_server.build_agent", side_effect=Exception("engine error")):
        response = client.post("/query", json={"question": "What is 42?"})

    assert response.status_code == 500
    assert "error" in response.json()


def test_get_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RAG Search" in response.text


def test_post_query_returns_snippets():
    mock_agent = MagicMock()
    mock_agent.chat.return_value = _make_mock_agent_response()

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=["docs"]), \
         patch("mcp_server.build_agent", return_value=mock_agent):
        response = client.post("/query", json={"question": "What is 42?"})

    assert response.status_code == 200
    data = response.json()
    assert "snippets" in data
    assert isinstance(data["snippets"], list)
    assert len(data["snippets"]) == 1
    assert data["snippets"][0]["filename"] == "doc.pdf"
    assert "text" in data["snippets"][0]


def test_post_query_stream_returns_event_sequence():
    mock_agent = MagicMock()
    mock_agent.stream_chat.return_value = _make_mock_streaming_response()

    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=["docs"]), \
         patch("mcp_server.build_agent", return_value=mock_agent):
        response = client.post("/query/stream", json={"question": "What is 42?"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = [
        json.loads(line[6:])
        for line in response.text.split("\n\n")
        if line.startswith("data: ")
    ]

    token_texts = [e["text"] for e in events if e["type"] == "token"]
    assert token_texts == ["The ", "answer ", "is 42."]

    sources_events = [e for e in events if e["type"] == "sources"]
    assert len(sources_events) == 1
    assert sources_events[0]["snippets"][0]["filename"] == "doc.pdf"

    assert events[-1]["type"] == "done"


def test_post_query_stream_error_when_ollama_not_running():
    with patch("mcp_server.check_ollama_running", return_value=False):
        response = client.post("/query/stream", json={"question": "hi"})

    events = [
        json.loads(line[6:])
        for line in response.text.split("\n\n")
        if line.startswith("data: ")
    ]
    assert events[0]["type"] == "error"
    assert "Ollama" in events[0]["message"]


def test_post_query_stream_error_when_no_indexes():
    with patch("mcp_server.check_ollama_running", return_value=True), \
         patch("mcp_server.list_indexed", return_value=[]):
        response = client.post("/query/stream", json={"question": "hi"})

    events = [
        json.loads(line[6:])
        for line in response.text.split("\n\n")
        if line.startswith("data: ")
    ]
    assert events[0]["type"] == "error"
    assert "No indexes" in events[0]["message"]
