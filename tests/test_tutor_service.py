import pytest
from unittest.mock import AsyncMock, MagicMock
from shared.interfaces import VectorStore, MessageRepository, GapRepository, StudentProfileRepository, Embedder, ChatModel, TutorResponse
from tutor.service import TutorService


@pytest.mark.asyncio
async def test_tutor_service_grounded_answer():
    mock_vs = AsyncMock(spec=VectorStore)
    mock_vs.search_chunks.return_value = [
        {"chunk_id": "c1", "filename": "physics.pdf", "content": "Newton's second law is F = ma."}
    ]
    mock_msg_repo = AsyncMock(spec=MessageRepository)
    mock_msg_repo.get_history.return_value = []
    
    mock_gap_repo = AsyncMock(spec=GapRepository)
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.embed.return_value = [0.1] * 1536
    
    mock_chat = MagicMock(spec=ChatModel)
    mock_chat.complete.return_value = TutorResponse(
        grounded=True,
        answer="According to [Chunk: c1], F = ma.",
        gaps=[],
        cited_chunk_ids=["c1"]
    )

    service = TutorService(
        vector_store=mock_vs,
        message_repo=mock_msg_repo,
        gap_repo=mock_gap_repo,
        embedder=mock_embedder,
        chat_model=mock_chat
    )

    result = await service.ask_tutor("phys-101", "student-1", "What is Newton's second law?")

    assert result.supplementary_answer is None
    assert "F = ma" in result.grounded_answer
    assert result.cited_chunk_ids == ["c1"]
    assert mock_msg_repo.save_message.call_count == 2  # user & assistant saved


@pytest.mark.asyncio
async def test_greeting_short_circuit_no_llm_or_embedder_calls():
    mock_vs = AsyncMock(spec=VectorStore)
    mock_msg_repo = AsyncMock(spec=MessageRepository)
    mock_gap_repo = AsyncMock(spec=GapRepository)
    mock_embedder = MagicMock(spec=Embedder)
    mock_chat = MagicMock(spec=ChatModel)

    service = TutorService(
        vector_store=mock_vs,
        message_repo=mock_msg_repo,
        gap_repo=mock_gap_repo,
        embedder=mock_embedder,
        chat_model=mock_chat
    )

    result = await service.ask_tutor("phys-101", "student-1", "hello")

    # Assert 0 calls to embedder and chat_model
    assert mock_embedder.embed.call_count == 0
    assert mock_chat.complete.call_count == 0
    assert mock_vs.search_chunks.call_count == 0
    assert "introduce yourself" in result.grounded_answer
    assert result.supplementary_answer is None
    assert mock_msg_repo.save_message.call_count == 2


@pytest.mark.asyncio
async def test_pass2_invoked_when_not_grounded():
    mock_vs = AsyncMock(spec=VectorStore)
    mock_vs.search_chunks.return_value = []
    mock_msg_repo = AsyncMock(spec=MessageRepository)
    mock_msg_repo.get_history.return_value = []
    mock_gap_repo = AsyncMock(spec=GapRepository)
    
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.embed.return_value = [0.2] * 1536
    
    mock_chat = MagicMock(spec=ChatModel)
    # Pass 1 returns un-grounded TutorResponse, Pass 2 returns general knowledge text
    mock_chat.complete.side_effect = [
        TutorResponse(grounded=False, answer="", gaps=["Quantum Computing"], cited_chunk_ids=[]),
        "Quantum computing relies on qubits to perform superposition and entanglement computations."
    ]

    service = TutorService(
        vector_store=mock_vs,
        message_repo=mock_msg_repo,
        gap_repo=mock_gap_repo,
        embedder=mock_embedder,
        chat_model=mock_chat
    )

    result = await service.ask_tutor("phys-101", "student-1", "Explain quantum computing in detail.")

    assert mock_chat.complete.call_count == 2
    # Verify Pass 2 was called with max_tokens=150
    pass2_call_args = mock_chat.complete.call_args_list[1]
    assert pass2_call_args.kwargs.get("max_tokens") == 150

    assert result.grounded_answer == ""
    assert "qubits" in (result.supplementary_answer or "")
    assert "GROUNDING_GAP" not in (result.supplementary_answer or "")
    assert result.gaps_addressed == ["Quantum Computing"]
    mock_gap_repo.cache_gap.assert_called_once()


@pytest.mark.asyncio
async def test_student_name_persistence_and_personalized_greeting():
    mock_vs = AsyncMock(spec=VectorStore)
    mock_msg_repo = AsyncMock(spec=MessageRepository)
    mock_gap_repo = AsyncMock(spec=GapRepository)
    mock_profile_repo = AsyncMock(spec=StudentProfileRepository)
    mock_embedder = MagicMock(spec=Embedder)
    mock_chat = MagicMock(spec=ChatModel)

    # Initial state: no name saved
    mock_profile_repo.get_profile.return_value = None

    service = TutorService(
        vector_store=mock_vs,
        message_repo=mock_msg_repo,
        gap_repo=mock_gap_repo,
        embedder=mock_embedder,
        chat_model=mock_chat,
        profile_repo=mock_profile_repo
    )

    # Step 1: Student introduces themselves
    res1 = await service.ask_tutor("cs-101", "student-42", "My name is Martins")
    assert "Martins" in res1.grounded_answer
    mock_profile_repo.set_preferred_name.assert_called_with("student-42", "Martins")

    # Step 2: Simulate subsequent profile fetch returning saved name
    mock_profile_repo.get_profile.return_value = {"student_id": "student-42", "preferred_name": "Martins"}

    # Step 3: Student greets again
    res2 = await service.ask_tutor("cs-101", "student-42", "hello")
    assert "Hello Martins!" in res2.grounded_answer


@pytest.mark.asyncio
async def test_student_name_update():
    mock_vs = AsyncMock(spec=VectorStore)
    mock_msg_repo = AsyncMock(spec=MessageRepository)
    mock_gap_repo = AsyncMock(spec=GapRepository)
    mock_profile_repo = AsyncMock(spec=StudentProfileRepository)
    mock_embedder = MagicMock(spec=Embedder)
    mock_chat = MagicMock(spec=ChatModel)

    mock_profile_repo.get_profile.return_value = {"student_id": "student-42", "preferred_name": "Martins"}

    service = TutorService(
        vector_store=mock_vs,
        message_repo=mock_msg_repo,
        gap_repo=mock_gap_repo,
        embedder=mock_embedder,
        chat_model=mock_chat,
        profile_repo=mock_profile_repo
    )

    # Student updates name
    res = await service.ask_tutor("cs-101", "student-42", "Call me Alex now")
    assert "Alex" in res.grounded_answer
    mock_profile_repo.set_preferred_name.assert_called_with("student-42", "Alex")

