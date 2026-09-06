import pytest
from unittest.mock import patch, MagicMock
from config import settings
from shared.dependencies import get_embedder, get_chat_model
from tutor.llm import OpenAIEmbedder, GeminiEmbedder, OpenAIChatModel, GeminiChatModel, MockEmbedder
from tutor.core import eval_grounding


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [True, False])
async def test_mixed_citations_are_cleaned_in_answer_and_history(structured):
    from unittest.mock import AsyncMock
    from shared.interfaces import TutorResponse
    from tutor.service import TutorService

    vectors, messages, gaps = AsyncMock(), AsyncMock(), AsyncMock()
    vectors.search_chunks.return_value = [
        {"chunk_id": "real-id", "content": "Entropy describes energy dispersal.", "similarity": 0.8}
    ]
    messages.get_history.return_value = []
    embedder, chat = MagicMock(), MagicMock()
    embedder.embed.return_value = [0.1] * 1536
    text = "Entropy [Chunk: real-id]. Another claim [Chunk: invented]."
    chat.complete.return_value = TutorResponse(
        grounded=True, answer=text, gaps=[], cited_chunk_ids=["real-id", "invented"]
    ) if structured else text

    answer = await TutorService(vectors, messages, gaps, embedder, chat).ask_tutor(
        "CS101", "student-1", "Explain entropy."
    )

    assert "[Chunk: real-id]" in answer.grounded_answer
    assert "invented" not in answer.grounded_answer
    assert answer.cited_chunk_ids == ["real-id"]
    saved = messages.save_message.call_args.kwargs
    assert saved["content"] == answer.grounded_answer
    assert saved["sources"] == ["real-id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("explanation", [
    "NameError: a variable name is not defined.",
    "TypeError: this operation uses an incompatible type.",
    "Error: this is a sample application error message.",
])
async def test_programming_error_explanations_are_returned_and_cached(explanation):
    from unittest.mock import AsyncMock
    from shared.interfaces import TutorResponse
    from tutor.service import TutorService

    vectors, messages, gaps = AsyncMock(), AsyncMock(), AsyncMock()
    vectors.search_chunks.return_value = []
    messages.get_history.return_value = []
    embedder, chat = MagicMock(), MagicMock()
    embedder.embed.return_value = [0.1] * 1536
    chat.complete.side_effect = [
        TutorResponse(grounded=False, answer="", gaps=["Python errors"], cited_chunk_ids=[]),
        explanation,
    ]

    answer = await TutorService(vectors, messages, gaps, embedder, chat).ask_tutor(
        "CS101", "student-1", "Explain Python errors."
    )

    assert answer.supplementary_answer == explanation
    assert gaps.cache_gap.call_args.kwargs["supplementary_answer"] == explanation
    assert messages.save_message.call_args.kwargs["content"] == explanation


def test_provider_selection_reads_settings():
    with patch.object(settings, "OPENAI_API_KEY", "sk-proj-test12345"), \
         patch.object(settings, "GEMINI_API_KEY", ""):
        embedder = get_embedder()
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.api_key == "sk-proj-test12345"

        chat_model = get_chat_model()
        assert isinstance(chat_model, OpenAIChatModel)
        assert chat_model.api_key == "sk-proj-test12345"


def test_provider_failure_raises_runtime_error():
    embedder = OpenAIEmbedder(api_key="sk-proj-fake-key")
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("OpenAI API rate limit / 401 Unauthorized")
        mock_openai.return_value = mock_client

        with pytest.raises(RuntimeError) as exc_info:
            embedder.embed("test text")
        
        assert "OpenAI embedding provider error" in str(exc_info.value)
        assert "401 Unauthorized" in str(exc_info.value)


def test_grounding_eval_rejects_error_and_invented_citation():
    # 1. Error message check
    is_grounded, gap_topic, citations = eval_grounding("[LLM Error: Connection refused]", [{"chunk_id": "c1"}])
    assert is_grounded is False
    assert gap_topic == "LLM provider error"

    # 2. Invented citation check
    is_grounded, gap_topic, citations = eval_grounding("Answer [Chunk: hallucinated-id]", [{"chunk_id": "real-chunk-1"}])
    assert is_grounded is False
    assert citations == []

    # 3. Valid citation check
    is_grounded, gap_topic, citations = eval_grounding("Answer [Chunk: real-chunk-1]", [{"chunk_id": "real-chunk-1"}])
    assert is_grounded is True
    assert citations == ["real-chunk-1"]


def test_structured_model_invented_citation_rejected():
    from shared.interfaces import TutorResponse
    from tutor.service import validate_tutor_response

    # Structured model returns grounded=True with an invented chunk ID
    structured_resp = TutorResponse(
        grounded=True,
        answer="Quantum supremacy achieved.",
        gaps=[],
        cited_chunk_ids=["invented-chunk-999"]
    )
    context_chunks = [{"chunk_id": "real-chunk-1"}]

    validated = validate_tutor_response(structured_resp, context_chunks)
    assert validated.grounded is False
    assert validated.cited_chunk_ids == []
    assert validated.answer == ""


@pytest.mark.asyncio
async def test_pass2_provider_error_not_cached_and_sanitized():
    from unittest.mock import AsyncMock
    from tutor.service import TutorService
    from shared.interfaces import TutorResponse

    mock_vstore = MagicMock()
    mock_vstore.search_chunks = AsyncMock(return_value=[])
    
    mock_mrepo = MagicMock()
    mock_mrepo.get_history = AsyncMock(return_value=[])
    mock_mrepo.save_message = AsyncMock(return_value=None)

    mock_grepo = MagicMock()
    mock_grepo.cache_gap = AsyncMock(return_value=None)

    mock_embedder = MagicMock()
    mock_embedder.embed = MagicMock(return_value=[0.1] * 1536)

    mock_chat = MagicMock()
    # Pass 1 returns ungrounded, Pass 2 returns LLM Error string
    mock_chat.complete.side_effect = [
        TutorResponse(grounded=False, answer="", gaps=["Complex Physics"], cited_chunk_ids=[]),
        "[LLM Error: Connection refused to Gemini API]"
    ]

    service = TutorService(
        vector_store=mock_vstore,
        message_repo=mock_mrepo,
        gap_repo=mock_grepo,
        embedder=mock_embedder,
        chat_model=mock_chat
    )

    ans = await service.ask_tutor("CS101", "student-1", "Explain string theory")

    # Verify provider error message is sanitized and NOT cached in gap_repo
    assert "[LLM Error:" not in (ans.supplementary_answer or "")
    assert ans.supplementary_answer == "The uploaded course materials for this course do not currently cover this specific question."
    mock_grepo.cache_gap.assert_not_called()


@pytest.mark.asyncio
async def test_low_similarity_produces_low_unfloored_confidence():
    from tutor.service import TutorService
    from shared.interfaces import TutorResponse

    mock_vstore = MagicMock()
    mock_mrepo = MagicMock()
    mock_grepo = MagicMock()
    mock_embedder = MagicMock()
    mock_chat = MagicMock()

    # Low similarity chunk (0.01)
    async def fake_search(*args, **kwargs): return [{"chunk_id": "c1", "similarity": 0.01}]
    async def fake_hist(*args, **kwargs): return []
    async def fake_save(*args, **kwargs): return None
    async def fake_cache(*args, **kwargs): return None
    mock_vstore.search_chunks = fake_search
    mock_mrepo.get_history = fake_hist
    mock_mrepo.save_message = fake_save
    mock_grepo.cache_gap = fake_cache

    mock_embedder.embed = MagicMock(return_value=[0.01] * 1536)
    mock_chat.complete.return_value = TutorResponse(grounded=True, answer="Derivative is 2x [Chunk: c1]", cited_chunk_ids=["c1"], gaps=[])

    service = TutorService(
        vector_store=mock_vstore,
        message_repo=mock_mrepo,
        gap_repo=mock_grepo,
        embedder=mock_embedder,
        chat_model=mock_chat
    )

    ans = await service.ask_tutor("MATH101", "student-1", "What is d/dx x^2?")
    assert ans.confidence == 0.01
