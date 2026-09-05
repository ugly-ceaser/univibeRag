import pytest
from tutor.prompts import format_context, build_tutor_messages, SYSTEM_PROMPT

def test_format_context_empty():
    res = format_context([])
    assert "No relevant course material" in res

def test_format_context_populated():
    chunks = [
        {"chunk_id": "chunk-1", "filename": "bio.pdf", "content": "Photosynthesis process..."},
        {"chunk_id": "chunk-2", "filename": "bio.pdf", "content": "Chlorophyll absorbs light..."}
    ]
    res = format_context(chunks)
    assert "[Chunk: chunk-1]" in res
    assert "Photosynthesis" in res

def test_build_tutor_messages():
    chunks = [{"chunk_id": "c1", "filename": "math.pdf", "content": "Calculus integration rule."}]
    history = [{"role": "user", "content": "What is calculus?"}, {"role": "assistant", "content": "It is math."}]
    question = "Explain integration."
    
    msgs = build_tutor_messages(question, chunks, history)
    
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert "Calculus integration rule" in msgs[1]["content"]
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == question
