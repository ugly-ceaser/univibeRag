import pytest
from shared.interfaces import TutorResponse
from tutor.core import eval_grounding, compose_response, is_greeting_or_smalltalk, extract_name_from_text


def test_eval_grounding_gap_detected():

    text = "GROUNDING_GAP: [Quantum Computing]\nThe uploaded course materials do not cover quantum computing."
    chunks = []
    is_grounded, gap_topic, citations = eval_grounding(text, chunks)
    
    assert is_grounded is False
    assert gap_topic == "Quantum Computing"
    assert citations == []


def test_eval_grounding_grounded_with_explicit_citations():
    text = "Based on [Chunk: abc-123] and [Chunk: def-456], derivative of x^2 is 2x."
    chunks = [{"chunk_id": "abc-123"}, {"chunk_id": "def-456"}]
    is_grounded, gap_topic, citations = eval_grounding(text, chunks)
    
    assert is_grounded is True
    assert gap_topic is None
    assert citations == ["abc-123", "def-456"]


def test_eval_grounding_fallback_citations():
    text = "The derivative of x^2 is 2x."
    chunks = [{"chunk_id": "chunk-99"}]
    is_grounded, gap_topic, citations = eval_grounding(text, chunks)
    
    assert is_grounded is True
    assert citations == []


def test_compose_response_never_leaks_grounding_gap():
    # Pass 1 response with gaps and Pass 2 text containing raw tags
    pass1 = TutorResponse(
        grounded=False,
        answer="GROUNDING_GAP: [Quantum Computing]",
        gaps=["Quantum Computing"],
        cited_chunk_ids=[]
    )
    pass2_text = "GROUNDING_GAP: [Quantum Computing]\nQuantum computing uses qubits to perform calculations."

    ans = compose_response(pass1, pass2_text)

    assert "GROUNDING_GAP" not in ans.grounded_answer
    assert "GROUNDING_GAP" not in (ans.supplementary_answer or "")
    assert ans.grounded_answer == ""
    assert "Quantum computing uses qubits" in ans.supplementary_answer
    assert ans.gaps_addressed == ["Quantum Computing"]


def test_is_greeting_or_smalltalk():
    assert is_greeting_or_smalltalk("hello") is True
    assert is_greeting_or_smalltalk("hi there") is True
    assert is_greeting_or_smalltalk("thanks") is True
    assert is_greeting_or_smalltalk("What is the time complexity of quicksort?") is False


def test_extract_name_from_text():
    assert extract_name_from_text("My name is Martins") == "Martins"
    assert extract_name_from_text("I am Alex") == "Alex"
    assert extract_name_from_text("Please call me Dr. Smith") == "Dr. Smith"
    assert extract_name_from_text("Call me Alexander now") == "Alexander"
    assert extract_name_from_text("What is my name?") is None
    assert extract_name_from_text("Explain Newton's second law") is None


def test_eval_grounding_rejects_llm_provider_error():
    text = "[LLM Error: Rate limit exceeded or API key invalid]"
    chunks = [{"chunk_id": "abc-123"}]
    is_grounded, gap_topic, citations = eval_grounding(text, chunks)

    assert is_grounded is False
    assert gap_topic == "LLM provider error"
    assert citations == []


def test_eval_grounding_rejects_invented_citations():
    text = "Here is the answer based on [Chunk: fake-invented-id]."
    chunks = [{"chunk_id": "real-chunk-100"}]
    is_grounded, gap_topic, citations = eval_grounding(text, chunks)

    assert is_grounded is False
    assert "Invented citation" in gap_topic
    assert citations == []


