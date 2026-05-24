from pathlib import Path
from unittest.mock import patch, MagicMock
from utils.image_describer import describe_image


def test_returns_string_description():
    mock_client = MagicMock()
    mock_client.generate.return_value = {"response": "A red circle on white background"}
    with patch("utils.image_describer.Client", return_value=mock_client), \
         patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"fake_image_bytes"
        result = describe_image(Path("fake_image.png"))
    assert result == "A red circle on white background"


def test_calls_llava_model():
    mock_client = MagicMock()
    mock_client.generate.return_value = {"response": "some description"}
    with patch("utils.image_describer.Client", return_value=mock_client), \
         patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"fake_image_bytes"
        describe_image(Path("fake.jpg"))
    call_kwargs = mock_client.generate.call_args[1]
    assert call_kwargs["model"] == "llava"


def test_passes_image_bytes_as_base64():
    mock_client = MagicMock()
    mock_client.generate.return_value = {"response": "desc"}
    with patch("utils.image_describer.Client", return_value=mock_client), \
         patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"bytes"
        describe_image(Path("img.png"))
    call_kwargs = mock_client.generate.call_args[1]
    assert len(call_kwargs["images"]) == 1
