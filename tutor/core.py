import re
from typing import Any
from shared.interfaces import TutorResponse, TutorAnswer


def is_greeting_or_smalltalk(question: str) -> bool:
    """
    Lightweight pre-check for greetings and trivial small talk.
    Short-circuits before vector search or LLM calls.
    """
    if not question or not question.strip():
        return True

    clean = question.strip().lower()
    clean_words = re.sub(r'[^\w\s]', '', clean).split()
    if not clean_words:
        return True

    greetings = {
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "thanks", "thankyou", "thank you", "thx", "help", "who are you"
    }

    if len(clean_words) <= 3 and any(w in greetings for w in clean_words):
        return True
    if len(clean_words) == 1 and clean_words[0] in greetings:
        return True

    return False


def extract_name_from_text(question: str) -> str | None:
    """
    Extracts a student's preferred name if the message introduces or updates their name.
    Returns the cleaned name string, or None if no name introduction is detected.
    """
    if not question or not question.strip():
        return None

    clean = question.strip()

    # Skip questions asking about names or generic queries
    if re.search(r'\b(what|who|is|can|do|does)\b.*\b(my name|call me)\b', clean, re.IGNORECASE):
        return None

    patterns = [
        r'\b(?:my name is|my name\'s)\s+([A-Za-z0-9_\-\.\'\s]+)',
        r'\b(?:you can call me|please call me|call me|address me as|refer to me as)\s+([A-Za-z0-9_\-\.\'\s]+)',
        r'\b(?:i am|i\'m)\s+([A-Z][a-zA-Z0-9_\-\.\'\s]*)',
        r'\b(?:i go by)\s+([A-Za-z0-9_\-\.\'\s]+)',
    ]

    invalid_words = {
        "a", "an", "the", "student", "user", "asking", "here", "ready", "new",
        "trying", "studying", "learning", "looking", "going", "doing", "fine", "ok", "good"
    }

    for pat in patterns:
        match = re.search(pat, clean, re.IGNORECASE)
        if match:
            raw_name = match.group(1).strip()
            # Protect common titles with period (e.g. Dr., Mr., Mrs., Ms., Prof., Sr., Jr.)
            protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.', r'\1___DOT___', raw_name, flags=re.IGNORECASE)
            parts = re.split(r'[,!\?\n]|\.\s+|\b(?:how|what|and|nice|hello|hi|please|from|now|today)\b', protected, maxsplit=1, flags=re.IGNORECASE)
            raw_name = parts[0].replace('___DOT___', '.').strip("'\" ").rstrip(".")
            if 2 <= len(raw_name) <= 40 and raw_name.lower() not in invalid_words:
                return raw_name

    return None



def sanitize_text(text: str | None) -> str:
    """Ensures internal tag GROUNDING_GAP never leaks into displayed user text."""
    if not text:
        return ""
    cleaned = re.sub(r'GROUNDING_GAP:\s*\[?[^\]\n]*\]?\s*', '', text).strip()
    return cleaned


def eval_grounding(response_text: str, context_chunks: list[dict[str, Any]]) -> tuple[bool, str | None, list[str]]:
    """
    Evaluates LLM response for grounding and citations.
    Returns:
      (is_grounded, gap_topic, cited_chunk_ids)
    """
    if "GROUNDING_GAP:" in response_text:
        match = re.search(r'GROUNDING_GAP:\s*\[?([^\]\n]+)\]?', response_text)
        gap_topic = match.group(1).strip() if match else "Unspecified topic"
        return False, gap_topic, []

    cited_ids = []
    chunk_pattern = re.compile(r'\[Chunk:\s*([a-f0-9\-]+)\]', re.IGNORECASE)
    matches = chunk_pattern.findall(response_text)
    for m in matches:
        if m not in cited_ids:
            cited_ids.append(m)

    if not cited_ids and context_chunks:
        cited_ids = [c["chunk_id"] for c in context_chunks[:3] if "chunk_id" in c]

    return True, None, cited_ids


def compose_response(
    pass1: TutorResponse | Any,
    pass2_text: str | None = None
) -> TutorAnswer:
    """
    Composes structured TutorAnswer ensuring GROUNDING_GAP never leaks into grounded_answer or supplementary_answer.
    """
    if isinstance(pass1, TutorResponse):
        grounded = pass1.grounded
        answer_text = sanitize_text(pass1.answer)
        gaps = pass1.gaps or []
        cited_chunk_ids = pass1.cited_chunk_ids or []
    elif isinstance(pass1, dict):
        grounded = bool(pass1.get("grounded", False))
        answer_text = sanitize_text(str(pass1.get("answer", "")))
        gaps = pass1.get("gaps", [])
        cited_chunk_ids = pass1.get("cited_chunk_ids", [])
    else:
        raw_str = str(pass1)
        grounded = "GROUNDING_GAP:" not in raw_str and bool(sanitize_text(raw_str))
        answer_text = sanitize_text(raw_str) if grounded else ""
        match = re.search(r'GROUNDING_GAP:\s*\[?([^\]\n]+)\]?', raw_str)
        gap_topic = match.group(1).strip() if match else "Unspecified topic"
        gaps = [gap_topic] if not grounded else []
        cited_chunk_ids = []

    if grounded:
        return TutorAnswer(
            grounded_answer=answer_text,
            supplementary_answer=None,
            cited_chunk_ids=cited_chunk_ids,
            gaps_addressed=[]
        )
    else:
        clean_pass2 = sanitize_text(pass2_text)
        if not clean_pass2:
            clean_pass2 = "The uploaded course materials for this course do not currently cover this specific question."
        return TutorAnswer(
            grounded_answer="",
            supplementary_answer=clean_pass2,
            cited_chunk_ids=[],
            gaps_addressed=gaps if gaps else ["Unspecified topic"]
        )
