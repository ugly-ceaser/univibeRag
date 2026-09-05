import re
from typing import Any
from shared.interfaces import (
    VectorStore, MessageRepository, GapRepository, StudentProfileRepository, Embedder, ChatModel, TutorResponse, TutorAnswer
)
from tutor.prompts import build_tutor_messages, build_pass2_messages
from tutor.core import is_greeting_or_smalltalk, extract_name_from_text, compose_response, eval_grounding

class TutorService:
    def __init__(
        self,
        vector_store: VectorStore,
        message_repo: MessageRepository,
        gap_repo: GapRepository,
        embedder: Embedder,
        chat_model: ChatModel,
        profile_repo: StudentProfileRepository | None = None
    ):
        self.vector_store = vector_store
        self.message_repo = message_repo
        self.gap_repo = gap_repo
        self.embedder = embedder
        self.chat_model = chat_model
        self.profile_repo = profile_repo

    async def ask_tutor(
        self,
        course_id: str,
        student_id: str,
        question: str
    ) -> TutorAnswer:
        # Load existing profile to check preferred name
        existing_profile = await self.profile_repo.get_profile(student_id) if self.profile_repo else None
        preferred_name = existing_profile.get("preferred_name") if existing_profile else None

        # Check if question provides or updates the student's preferred name
        extracted_name = extract_name_from_text(question)
        if extracted_name:
            if self.profile_repo:
                await self.profile_repo.set_preferred_name(student_id, extracted_name)
            name_reply = f"Nice to meet you, {extracted_name}! I've saved your name and will address you as {extracted_name} from now on. How can I help you today?"
            await self.message_repo.save_message(
                course_id=course_id,
                student_id=student_id,
                role="user",
                content=question
            )
            await self.message_repo.save_message(
                course_id=course_id,
                student_id=student_id,
                role="assistant",
                content=name_reply,
                grounded=True,
                sources=[]
            )
            return TutorAnswer(
                grounded_answer=name_reply,
                supplementary_answer=None,
                cited_chunk_ids=[],
                gaps_addressed=[]
            )

        # Pre-check: Greeting and trivial small talk short-circuit
        if is_greeting_or_smalltalk(question):
            if preferred_name:
                greeting_reply = f"Hello {preferred_name}! Welcome back. How can I help you today?"
            else:
                greeting_reply = "Hello! I'm your AI Tutor for this course. Welcome! Please introduce yourself so I know how to address you, and let me know what you'd like to learn or ask about today."
            await self.message_repo.save_message(
                course_id=course_id,
                student_id=student_id,
                role="user",
                content=question
            )
            await self.message_repo.save_message(
                course_id=course_id,
                student_id=student_id,
                role="assistant",
                content=greeting_reply,
                grounded=True,
                sources=[]
            )
            return TutorAnswer(
                grounded_answer=greeting_reply,
                supplementary_answer=None,
                cited_chunk_ids=[],
                gaps_addressed=[]
            )

        # Step 1: Embed question
        question_vector = self.embedder.embed(question)

        # Step 2: Retrieve relevant chunks & chat history
        context_chunks = await self.vector_store.search_chunks(question_vector, course_id=course_id, top_k=6)
        chat_history = await self.message_repo.get_history(course_id=course_id, student_id=student_id, limit=10)

        # Step 3: Pass 1 Completion
        messages = build_tutor_messages(question, context_chunks, chat_history, student_name=preferred_name)
        pass1_raw = self.chat_model.complete(messages, response_schema=TutorResponse)

        # Convert pass1_raw to TutorResponse
        if isinstance(pass1_raw, TutorResponse):
            pass1_res = pass1_raw
        elif isinstance(pass1_raw, dict):
            pass1_res = TutorResponse(
                grounded=bool(pass1_raw.get("grounded", False)),
                answer=str(pass1_raw.get("answer", "")),
                gaps=pass1_raw.get("gaps", []),
                cited_chunk_ids=pass1_raw.get("cited_chunk_ids", [])
            )
        else:
            raw_str = str(pass1_raw)
            is_grounded, gap_topic, citations = eval_grounding(raw_str, context_chunks)
            pass1_res = TutorResponse(
                grounded=is_grounded,
                answer=raw_str if is_grounded else "",
                gaps=[gap_topic] if (not is_grounded and gap_topic) else [],
                cited_chunk_ids=citations if is_grounded else []
            )

        # Step 4: Pass 2 Completion (fired when grounded=False)
        pass2_text = None
        if not pass1_res.grounded:
            gaps = pass1_res.gaps or [question[:50]]
            pass2_msgs = build_pass2_messages(question, gaps)
            pass2_raw = self.chat_model.complete(pass2_msgs, max_tokens=150)
            if isinstance(pass2_raw, dict):
                pass2_text = str(pass2_raw.get("content", pass2_raw))
            else:
                pass2_text = str(pass2_raw)

            # Cache gap
            gap_topic = gaps[0] if gaps else question[:50]
            await self.gap_repo.cache_gap(
                course_id=course_id,
                topic=gap_topic,
                topic_embedding=question_vector,
                supplementary_answer=pass2_text or ""
            )

        # Step 5: Compose structured answer
        final_answer = compose_response(pass1_res, pass2_text)

        # Save to message history
        await self.message_repo.save_message(
            course_id=course_id,
            student_id=student_id,
            role="user",
            content=question,
            embedding=question_vector
        )
        saved_content = final_answer.grounded_answer or final_answer.supplementary_answer or ""
        await self.message_repo.save_message(
            course_id=course_id,
            student_id=student_id,
            role="assistant",
            content=saved_content,
            grounded=bool(final_answer.grounded_answer),
            sources=final_answer.cited_chunk_ids
        )

        return final_answer
