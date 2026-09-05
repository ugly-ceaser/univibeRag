import os
import uuid
from pathlib import Path
from config import settings
from shared.interfaces import FileStorage

class LocalStorage(FileStorage):
    def __init__(self, base_dir: str = settings.STORAGE_DIR):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, filename: str, content: bytes) -> str:
        """Save file bytes locally and return relative storage path."""
        ext = Path(filename).suffix
        unique_name = f"{uuid.uuid4()}{ext}"
        target_path = self.base_dir / unique_name
        
        with open(target_path, "wb") as f:
            f.write(content)
            
        return str(target_path.relative_to(self.base_dir))

    def get_file(self, storage_path: str) -> bytes:
        """Read and return file bytes from storage path."""
        file_path = (self.base_dir / storage_path).resolve()
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found in storage: {storage_path}")
            
        with open(file_path, "rb") as f:
            return f.read()

    def delete_file(self, storage_path: str) -> bool:
        """Delete file from storage if present."""
        file_path = (self.base_dir / storage_path).resolve()
        if file_path.exists() and file_path.is_file():
            os.remove(file_path)
            return True
        return False
