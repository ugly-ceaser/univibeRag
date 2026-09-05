import pytest
from shared.interfaces import RawDocument, ChunkConfig
from ingestion.chunkers import HeaderAwareChunker

def test_header_aware_chunker_empty():
    chunker = HeaderAwareChunker()
    doc = RawDocument(filename="empty.txt", text_content="")
    chunks = chunker.chunk(doc)
    assert chunks == []

def test_header_aware_chunker_with_headers():
    chunker = HeaderAwareChunker()
    text = """
# Section 1: Introduction
This is the introduction to artificial intelligence and vector search.

## Section 2: Retrieval
Retrieval augmented generation uses embeddings to search relevant context.
"""
    doc = RawDocument(filename="sample.md", text_content=text)
    chunks = chunker.chunk(doc)
    
    assert len(chunks) >= 2
    assert "[Section 1: Introduction]" in chunks[0].text or "introduction" in chunks[0].text.lower()
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1

def test_header_aware_chunker_max_token_split():
    chunker = HeaderAwareChunker(default_config=ChunkConfig(max_tokens=20, overlap_tokens=5))
    long_text = "word " * 100
    doc = RawDocument(filename="long.txt", text_content=long_text)
    chunks = chunker.chunk(doc)
    
    assert len(chunks) > 1
    # Check each chunk length is reasonable
    for chunk in chunks:
        assert len(chunk.text.split()) <= 25
