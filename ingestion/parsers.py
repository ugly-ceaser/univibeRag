import io
import os
from typing import Any
from shared.interfaces import DocumentParser, RawDocument, OCREngine

class PDFParser(DocumentParser):
    def parse(self, file_bytes: bytes, mime_type: str, ocr_engine: OCREngine | None = None) -> RawDocument:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages_text: list[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{text}")

        combined_text = "\n\n".join(pages_text)
        if not combined_text.strip() and ocr_engine is not None:
            # Fallback to OCR if PDF contains images/no selectable text
            combined_text = ocr_engine.extract_text(file_bytes, mime_type)

        return RawDocument(
            filename="document.pdf",
            text_content=combined_text,
            metadata={"num_pages": len(reader.pages), "parser": "PDFParser"}
        )


class DocxParser(DocumentParser):
    def parse(self, file_bytes: bytes, mime_type: str, ocr_engine: OCREngine | None = None) -> RawDocument:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        combined_text = "\n\n".join(paragraphs)

        return RawDocument(
            filename="document.docx",
            text_content=combined_text,
            metadata={"num_paragraphs": len(paragraphs), "parser": "DocxParser"}
        )


class TextParser(DocumentParser):
    def parse(self, file_bytes: bytes, mime_type: str, ocr_engine: OCREngine | None = None) -> RawDocument:
        text = file_bytes.decode('utf-8', errors='ignore')
        return RawDocument(
            filename="document.txt",
            text_content=text,
            metadata={"parser": "TextParser"}
        )


class ImageParser(DocumentParser):
    def parse(self, file_bytes: bytes, mime_type: str, ocr_engine: OCREngine | None = None) -> RawDocument:
        text = ""
        if ocr_engine is not None:
            try:
                text = ocr_engine.extract_text(file_bytes, mime_type)
            except Exception as e:
                text = f"[OCR Failed: {str(e)}]"
        else:
            text = "[Image file uploaded - OCR engine not configured]"

        return RawDocument(
            filename="image.png",
            text_content=text,
            metadata={"parser": "ImageParser", "mime_type": mime_type}
        )


class PyTesseractOCREngine(OCREngine):
    """Implementation of OCREngine using pytesseract and PIL."""
    def extract_text(self, file_bytes: bytes, mime_type: str) -> str:
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            return f"[OCR Error: {str(e)}]"


def get_parser_for_filename(filename: str, mime_type: str = "") -> DocumentParser:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf" or "pdf" in mime_type:
        return PDFParser()
    elif ext in [".docx", ".doc"] or "word" in mime_type:
        return DocxParser()
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"] or "image" in mime_type:
        return ImageParser()
    else:
        return TextParser()
