import re
from typing import Any
from shared.interfaces import ChunkingStrategy, RawDocument, ChunkConfig, ChunkDraft

class HeaderAwareChunker(ChunkingStrategy):
    """
    Header-aware text chunker.
    Splits text while preserving markdown headers / section context and enforcing chunk size limits with overlap.
    """

    def __init__(self, default_config: ChunkConfig | None = None):
        self.default_config = default_config or ChunkConfig()

    def chunk(self, raw_doc: RawDocument, config: ChunkConfig | None = None) -> list[ChunkDraft]:
        cfg = config or self.default_config
        text = raw_doc.text_content.strip()
        if not text:
            return []

        # Approximate words per token (1 token ~ 0.75 words, or 1 token ~ 4 chars)
        max_words = int(cfg.max_tokens * 0.75)
        overlap_words = int(cfg.overlap_tokens * 0.75)
        if max_words <= 0:
            max_words = 300
        if overlap_words < 0:
            overlap_words = 0

        # Split document by headers if present
        header_pattern = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)
        split_parts = header_pattern.split(text)

        sections: list[tuple[str, str]] = []  # (current_header, section_text)
        current_header = raw_doc.metadata.get("title", "")

        idx = 0
        while idx < len(split_parts):
            part = split_parts[idx].strip()
            if not part:
                idx += 1
                continue

            if header_pattern.match(part):
                current_header = part.lstrip('#').strip()
                if idx + 1 < len(split_parts):
                    body = split_parts[idx + 1].strip()
                    sections.append((current_header, body))
                    idx += 2
                else:
                    idx += 1
            else:
                sections.append((current_header, part))
                idx += 1

        chunks: list[ChunkDraft] = []
        chunk_counter = 0

        for header, body in sections:
            words = body.split()
            if not words:
                continue

            start_idx = 0
            while start_idx < len(words):
                end_idx = min(start_idx + max_words, len(words))
                chunk_words = words[start_idx:end_idx]
                chunk_text = " ".join(chunk_words)

                if cfg.header_preserve and header:
                    chunk_text = f"[{header}]\n{chunk_text}"

                metadata = dict(raw_doc.metadata)
                if header:
                    metadata["section_header"] = header

                chunks.append(
                    ChunkDraft(
                        text=chunk_text,
                        chunk_index=chunk_counter,
                        metadata=metadata
                    )
                )
                chunk_counter += 1

                if end_idx >= len(words):
                    break

                # Advance window with overlap
                start_idx += max(1, max_words - overlap_words)

        return chunks
