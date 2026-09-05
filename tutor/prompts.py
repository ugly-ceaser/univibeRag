from typing import Any

SYSTEM_PROMPT = """You are UniVibe AI Tutor, an expert academic assistant for university students.

Your primary directive is STRICT GROUNDING in the provided course materials:
1. Base your answer ONLY on the course material context chunks provided below.
2. Include chunk citations using `[Chunk: <chunk_id>]` when referencing specific concepts from the context.
3. If the provided course materials DO NOT contain sufficient information to answer the question, do not invent grounded claims. State clearly what is missing in your response structure.
"""

PASS2_SYSTEM_PROMPT = """The student asked something not covered in their uploaded course materials. Answer it directly using your general knowledge — don't just say it's not covered. Keep the answer concise: 2-4 sentences, enough to be genuinely useful without going deep. This is a supplementary note, not the primary lesson."""


def format_context(context_chunks: list[dict[str, Any]]) -> str:
    if not context_chunks:
        return "No relevant course material context found."

    formatted = []
    for chunk in context_chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        filename = chunk.get("filename", "unknown")
        content = chunk.get("content", "")
        formatted.append(f"--- [Chunk: {chunk_id}] (Source: {filename}) ---\n{content}")

    return "\n\n".join(formatted)


def build_tutor_messages(
    question: str,
    context_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
    student_name: str | None = None
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if student_name:
        messages.append({
            "role": "system",
            "content": f"STUDENT PROFILE:\nThe student's preferred name is '{student_name}'. Address them by their preferred name when appropriate."
        })

    # Include formatted course context
    context_str = format_context(context_chunks)
    context_msg = f"COURSE MATERIAL CONTEXT:\n{context_str}"
    messages.append({"role": "system", "content": context_msg})

    # Include previous conversation history (last N messages)
    for msg in chat_history[-6:]:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    # Add current question
    messages.append({"role": "user", "content": question})

    return messages


def build_pass2_messages(
    question: str,
    gaps: list[str] | None = None
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": PASS2_SYSTEM_PROMPT}]
    if gaps:
        gap_info = ", ".join(gaps)
        messages.append({"role": "system", "content": f"Identified Coverage Gaps: {gap_info}"})
    messages.append({"role": "user", "content": question})
    return messages
