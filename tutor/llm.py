import os
import random
from typing import Any
from shared.interfaces import ChatModel, Embedder, TutorResponse, ProviderUnavailableError

class MockEmbedder(Embedder):
    """Deterministic mock embedder generating 1536-dim normalized vectors for testing."""
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        # Hash text to seed random numbers deterministically
        seed = sum(ord(c) for c in text)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]
        norm = sum(x*x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class MockChatModel(ChatModel):
    """Mock LLM response generator for testing RAG flows."""
    def complete(
        self,
        messages: list[dict[str, Any]],
        response_schema: type | None = None,
        max_tokens: int | None = None
    ) -> Any:
        user_msg = ""
        context_msg = ""
        is_pass2 = False
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")
            elif m.get("role") == "system":
                content = m.get("content", "")
                if "COURSE MATERIAL CONTEXT" in content:
                    context_msg = content
                elif "uploaded course materials" in content or "supplementary note" in content:
                    is_pass2 = True

        if is_pass2:
            return f"Here is a concise supplementary answer regarding '{user_msg}': This concept represents standard academic knowledge in the field."

        if response_schema == TutorResponse or response_schema is not None:
            if "No relevant course material context found." in context_msg or not context_msg.strip():
                return TutorResponse(
                    grounded=False,
                    answer="",
                    gaps=[user_msg[:50] if user_msg else "Unspecified topic"],
                    cited_chunk_ids=[]
                )
            return TutorResponse(
                grounded=True,
                answer=f"Based on the provided course material [Chunk: sample-chunk-1], here is the answer to '{user_msg}': The context explains that this concept is fundamental to the course curriculum.",
                gaps=[],
                cited_chunk_ids=["sample-chunk-1"]
            )

        if "No relevant course material context found." in context_msg or not context_msg.strip():
            return TutorResponse(
                grounded=False,
                answer="",
                gaps=[user_msg[:50] if user_msg else "Unspecified topic"],
                cited_chunk_ids=[]
            )

        return f"Based on the provided course material [Chunk: sample-chunk-1], here is the answer to '{user_msg}': The context explains that this concept is fundamental to the course curriculum."


class GeminiChatModel(ChatModel):
    """Production ChatModel implementation backed by Google Gemini SDK."""
    def __init__(self, api_key: str | None = None, model_name: str = "gemini-1.5-flash"):
        from config import settings
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name

    def complete(
        self,
        messages: list[dict[str, Any]],
        response_schema: type | None = None,
        max_tokens: int | None = None
    ) -> Any:
        if not self.api_key:
            # Fall back to MockChatModel if API key not supplied
            return MockChatModel().complete(messages, response_schema=response_schema, max_tokens=max_tokens)

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            gen_config = {}
            if max_tokens:
                gen_config["max_output_tokens"] = max_tokens

            model = genai.GenerativeModel(self.model_name, generation_config=gen_config if gen_config else None)

            prompt_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
            response = model.generate_content(prompt_text)
            return response.text
        except Exception as e:
            return f"[LLM Error: {str(e)}]"


class OpenAIChatModel(ChatModel):
    """Production ChatModel implementation backed by OpenAI SDK."""
    def __init__(self, api_key: str | None = None, model_name: str = "gpt-4o-mini"):
        from config import settings
        self.api_key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        self.model_name = settings.LLM_MODEL or model_name

    def complete(
        self,
        messages: list[dict[str, Any]],
        response_schema: type | None = None,
        max_tokens: int | None = None
    ) -> Any:
        if not self.api_key:
            return MockChatModel().complete(messages, response_schema=response_schema, max_tokens=max_tokens)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            if response_schema == TutorResponse or (response_schema is not None and hasattr(response_schema, "__annotations__")):
                try:
                    response = client.beta.chat.completions.parse(
                        response_format=response_schema,
                        **kwargs
                    )
                    return response.choices[0].message.parsed
                except Exception:
                    pass

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM Error: {str(e)}]"


class OpenAIEmbedder(Embedder):
    """Production Embedder backed by OpenAI text-embedding-3-small."""
    def __init__(self, api_key: str | None = None):
        from config import settings
        self.api_key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            return MockEmbedder().embed_batch(texts)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            res = client.embeddings.create(input=texts, model="text-embedding-3-small")
            return [data.embedding for data in res.data]
        except Exception as e:
            raise ProviderUnavailableError(f"OpenAI embedding provider error: {str(e)}") from e


class GeminiEmbedder(Embedder):
    """Production Embedder backed by Google Gemini text-embedding-004."""
    def __init__(self, api_key: str | None = None):
        from config import settings
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            return MockEmbedder().embed_batch(texts)

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=texts
            )
            return result['embedding']
        except Exception as e:
            raise ProviderUnavailableError(f"Gemini embedding provider error: {str(e)}") from e

