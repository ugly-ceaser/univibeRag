import pytest
from unittest.mock import MagicMock
from ingestion.parsers import (
    TextParser,
    ImageParser,
    PDFParser,
    DocxParser,
    get_parser_for_filename
)
from shared.interfaces import OCREngine

def test_text_parser():
    parser = TextParser()
    content = "Hello, world! This is a plain text file."
    doc = parser.parse(content.encode("utf-8"), "text/plain")
    assert doc.text_content == content
    assert doc.metadata["parser"] == "TextParser"

def test_image_parser_with_ocr_engine():
    mock_ocr = MagicMock(spec=OCREngine)
    mock_ocr.extract_text.return_value = "Extracted text from image OCR"
    
    parser = ImageParser()
    doc = parser.parse(b"fake_image_bytes", "image/png", ocr_engine=mock_ocr)
    
    assert doc.text_content == "Extracted text from image OCR"
    mock_ocr.extract_text.assert_called_once_with(b"fake_image_bytes", "image/png")

def test_image_parser_without_ocr_engine():
    parser = ImageParser()
    doc = parser.parse(b"fake_image_bytes", "image/png", ocr_engine=None)
    assert doc.text_content == ""

def test_get_parser_for_filename():
    assert isinstance(get_parser_for_filename("doc.pdf"), PDFParser)
    assert isinstance(get_parser_for_filename("doc.docx"), DocxParser)
    assert isinstance(get_parser_for_filename("photo.png"), ImageParser)
    assert isinstance(get_parser_for_filename("notes.txt"), TextParser)
    assert isinstance(get_parser_for_filename("file_without_ext", mime_type="application/pdf"), PDFParser)
