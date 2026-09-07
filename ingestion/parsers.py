import io
import os
import base64
import logging
from typing import Any
from shared.interfaces import DocumentParser, RawDocument, OCREngine

logger = logging.getLogger(__name__)

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
            # Fallback to OCR if PDF contains scanned images/no selectable text
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
                logger.warning(f"OCR parsing exception: {str(e)}")
                text = ""

        if not text.strip() or text.startswith("[OCR"):
            text = ""

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
            logger.info(f"PyTesseract OCR fallback: {str(e)}")
            return ""


class GeminiVisionOCREngine(OCREngine):
    """Vision-based OCR using Google Gemini multimodal model."""
    def __init__(self, api_key: str | None = None):
        from config import settings
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")

    def extract_text(self, file_bytes: bytes, mime_type: str) -> str:
        if not self.api_key:
            return ""
        try:
            import google.generativeai as genai
            from PIL import Image

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            image = Image.open(io.BytesIO(file_bytes))
            prompt = (
                "Extract and transcribe all text, handwritten notes, tables, mathematical equations, "
                "and key visual content from this course material image accurately and verbatim."
            )
            response = model.generate_content([image, prompt])
            return (response.text or "").strip()
        except Exception as e:
            logger.warning(f"Gemini Vision OCR failed: {str(e)}")
            return ""


class OpenAIVisionOCREngine(OCREngine):
    """Vision-based OCR using OpenAI GPT-4o vision capability."""
    def __init__(self, api_key: str | None = None):
        from config import settings
        self.api_key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")

    def extract_text(self, file_bytes: bytes, mime_type: str) -> str:
        if not self.api_key:
            return ""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            b64_image = base64.b64encode(file_bytes).decode('utf-8')
            media_type = mime_type if mime_type.startswith("image/") else "image/png"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract and transcribe all text, formulas, and written content from this image verbatim."},
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64_image}"}}
                        ]
                    }
                ],
                max_tokens=1500
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"OpenAI Vision OCR failed: {str(e)}")
            return ""


class CompositeOCREngine(OCREngine):
    """Tries Tesseract OCR first, then falls back to Gemini Vision or OpenAI Vision."""
    def __init__(self, engines: list[OCREngine] | None = None):
        if engines is not None:
            self.engines = engines
        else:
            self.engines = [
                PyTesseractOCREngine(),
                GeminiVisionOCREngine(),
                OpenAIVisionOCREngine()
            ]

    def extract_text(self, file_bytes: bytes, mime_type: str) -> str:
        for engine in self.engines:
            try:
                extracted = engine.extract_text(file_bytes, mime_type)
                if extracted and extracted.strip() and not extracted.startswith("[OCR"):
                    logger.info(f"Successfully extracted OCR text using {engine.__class__.__name__}")
                    return extracted.strip()
            except Exception as e:
                logger.debug(f"Engine {engine.__class__.__name__} failed: {str(e)}")
        return ""


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
