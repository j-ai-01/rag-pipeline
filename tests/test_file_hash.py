import tempfile
from pathlib import Path
from utils.file_hash import file_hash


def test_returns_string():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"hello world")
        path = Path(f.name)
    assert isinstance(file_hash(path), str)


def test_same_content_same_hash():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"same content")
        path1 = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"same content")
        path2 = Path(f.name)
    assert file_hash(path1) == file_hash(path2)


def test_different_content_different_hash():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"content A")
        path1 = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"content B")
        path2 = Path(f.name)
    assert file_hash(path1) != file_hash(path2)
