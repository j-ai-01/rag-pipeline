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
