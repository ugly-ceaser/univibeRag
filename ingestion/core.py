import hashlib
from shared.interfaces import RawDocument, ChunkingStrategy, ChunkConfig, ChunkDraft, ChunkWithVector, Embedder

def hash_bytes(data: bytes) -> str:
    """Computes SHA-256 hex digest of file bytes for content hashing."""
    return hashlib.sha256(data).hexdigest()

def chunk_document(raw_doc: RawDocument, chunker: ChunkingStrategy, config: ChunkConfig | None = None) -> list[ChunkDraft]:
    """Pure pipeline step: breaks RawDocument into ChunkDraft items using the given ChunkingStrategy."""
    return chunker.chunk(raw_doc, config=config)

def attach_vectors(chunks: list[ChunkDraft], embedder: Embedder) -> list[ChunkWithVector]:
    """Pure pipeline step: computes vector embeddings for each chunk draft and pairs them up."""
    if not chunks:
        return []

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_batch(texts)

    result: list[ChunkWithVector] = []
    for chunk, emb in zip(chunks, embeddings):
        result.append(
            ChunkWithVector(
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                embedding=emb,
                metadata=chunk.metadata
            )
        )
    return result
