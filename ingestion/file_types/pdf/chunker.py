from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import List, Optional

from docling.chunking import HybridChunker


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page: int
    section: str = ""


class BaseChunker(ABC):
    @abstractmethod
    def chunk_document(self, document) -> list:
        raise NotImplementedError


class DoclingChunker(BaseChunker):
    """Docling HybridChunker preserving layout hierarchies and section headings."""
    def __init__(self, tokenizer=None, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens
        kwargs = {}
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        self.chunker = HybridChunker(**kwargs)

    def chunk_document(self, document) -> list:
        chunks = []
        for index, chunk in enumerate(self.chunker.chunk(document)):
            headings = getattr(chunk.meta, "headings", None) or []
            chunks.append(Chunk(
                text=self._text_of(chunk, headings),
                chunk_index=index,
                page=self._page_of(chunk),
                section=headings[-1] if headings else "",
            ))
        return chunks

    def _text_of(self, chunk, headings: list) -> str:
        if headings:
            return "\n".join(headings) + "\n" + chunk.text
        return chunk.text

    def _page_of(self, chunk) -> int:
        try:
            return chunk.meta.doc_items[0].prov[0].page_no
        except Exception:
            return 0


class RecursiveCharacterChunker(BaseChunker):
    """Recursively splits document text using hierarchical separators to maintain target chunk size."""
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_document(self, document) -> list:
        if hasattr(document, "export_to_markdown"):
            text = document.export_to_markdown()
        elif hasattr(document, "export_to_text"):
            text = document.export_to_text()
        else:
            text = str(document)

        raw_chunks = self._split_text(text, self.separators)
        chunks = []
        for idx, piece in enumerate(raw_chunks):
            clean = piece.strip()
            if not clean:
                continue
            chunks.append(Chunk(
                text=clean,
                chunk_index=idx,
                page=0,
                section="",
            ))
        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []
        _separator = "" if separator == "" else separator

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, _separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

        if good_splits:
            merged = self._merge_splits(good_splits, _separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0
        for s in splits:
            _len = len(s)
            if total + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc)
                    # Keep overlap if possible
                    while total > self.chunk_overlap or (
                        total + _len + (len(separator) if current_doc else 0) > self.chunk_size
                        and total > 0
                    ):
                        popped = current_doc.pop(0)
                        total -= len(popped) + (len(separator) if current_doc else 0)
                current_doc.append(s)
                total += _len + (len(separator) if len(current_doc) > 1 else 0)
            else:
                current_doc.append(s)
                total += _len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc)
        return docs


class FixedTokenChunker(BaseChunker):
    """Splits text by token or word approximation into fixed windows with specified overlap."""
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document) -> list:
        if hasattr(document, "export_to_markdown"):
            text = document.export_to_markdown()
        elif hasattr(document, "export_to_text"):
            text = document.export_to_text()
        else:
            text = str(document)

        words = re.findall(r"\S+|\n", text)
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        idx = 0
        for start in range(0, len(words), step):
            window = words[start: start + self.chunk_size]
            piece = " ".join(window).replace(" \n ", "\n").replace("\n ", "\n").strip()
            if piece:
                chunks.append(Chunk(
                    text=piece,
                    chunk_index=idx,
                    page=0,
                    section="",
                ))
                idx += 1
            if start + self.chunk_size >= len(words):
                break
        return chunks


def get_chunker(
    chunker_type: str = "docling_hybrid",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs,
) -> BaseChunker:
    """Factory helper to instantiate a BaseChunker by name and hyperparameters."""
    ctype = chunker_type.strip().lower()
    if ctype in ("docling", "docling_hybrid", "hybrid"):
        return DoclingChunker(**kwargs)
    elif ctype in ("recursive", "recursive_character", "character"):
        return RecursiveCharacterChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
        )
    elif ctype in ("fixed", "fixed_token", "token", "words"):
        return FixedTokenChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Unknown chunker_type '{chunker_type}'. Choose from: docling_hybrid, recursive, fixed_token")



