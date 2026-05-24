from unittest.mock import patch, MagicMock
import httpx
from utils.ollama_check import check_ollama_running, assert_ollama_running


def test_returns_true_when_ollama_responds():
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch("utils.ollama_check.httpx.get", return_value=mock_response):
        assert check_ollama_running() is True


def test_returns_false_when_connection_refused():
    with patch("utils.ollama_check.httpx.get", side_effect=httpx.ConnectError("refused")):
        assert check_ollama_running() is False


def test_assert_raises_system_exit_when_not_running():
    with patch("utils.ollama_check.check_ollama_running", return_value=False):
        try:
            assert_ollama_running()
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert "ollama serve" in str(e)


def test_assert_does_not_raise_when_running():
    with patch("utils.ollama_check.check_ollama_running", return_value=True):
        assert_ollama_running()  # should not raise
