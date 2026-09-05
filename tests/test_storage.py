import pytest
from pathlib import Path
from core.storage import LocalStorage

def test_storage_save_and_retrieve(tmp_path):
    """Verify that LocalStorage saves file bytes and retrieves exact binary content."""
    storage = LocalStorage(base_dir=str(tmp_path))
    content = b"Lecture slides content binary PDF"
    
    path = storage.save_file("lecture1.pdf", content)
    assert isinstance(path, str)
    assert path.endswith(".pdf")
    
    retrieved = storage.get_file(path)
    assert retrieved == content

def test_storage_get_nonexistent_file_raises_error(tmp_path):
    """Verify that attempting to read a non-existent file raises FileNotFoundError."""
    storage = LocalStorage(base_dir=str(tmp_path))
    
    with pytest.raises(FileNotFoundError):
        storage.get_file("non_existent_file.pdf")

def test_storage_delete_file(tmp_path):
    """Verify that deleting a stored file removes it from disk and returns True."""
    storage = LocalStorage(base_dir=str(tmp_path))
    path = storage.save_file("test.txt", b"sample data")
    
    assert storage.delete_file(path) is True
    
    # Second delete returns False because file is gone
    assert storage.delete_file(path) is False
    
    with pytest.raises(FileNotFoundError):
        storage.get_file(path)
