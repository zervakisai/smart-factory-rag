"""
Document Indexer — Chunk, embed, and index manufacturing documents.

Supports PDF, DOCX, Excel spec sheets with multilingual (EL/EN) awareness.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A document chunk with metadata for traceability."""

    text: str
    document: str
    page: int
    paragraph: int
    language: str  # "el" or "en"
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "chunk_text": self.text,
            "document": self.document,
            "page": self.page,
            "paragraph": self.paragraph,
            "language": self.language,
            **self.metadata,
        }


class DocumentChunker:
    """
    Intelligent document chunking with manufacturing-domain awareness.

    - Respects section boundaries (headers, tables, lists)
    - Preserves table structure for spec sheets
    - Language-aware splitting for Greek/English docs
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, file_path: Path) -> list[Chunk]:
        """Route document to appropriate chunker based on file type."""
        suffix = file_path.suffix.lower()
        handlers = {
            ".pdf": self._chunk_pdf,
            ".docx": self._chunk_docx,
            ".xlsx": self._chunk_excel,
            ".xls": self._chunk_excel,
            ".txt": self._chunk_text,
            ".md": self._chunk_text,
        }
        handler = handlers.get(suffix)
        if handler is None:
            logger.warning(f"Unsupported file type: {suffix} — skipping {file_path}")
            return []
        return handler(file_path)

    def _chunk_pdf(self, file_path: Path) -> list[Chunk]:
        """Extract and chunk PDF with page-level tracking."""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        chunks = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) < self.min_chunk_size:
                continue

            page_chunks = self._split_text(text)
            for para_idx, chunk_text in enumerate(page_chunks):
                chunks.append(Chunk(
                    text=chunk_text,
                    document=file_path.name,
                    page=page_num,
                    paragraph=para_idx + 1,
                    language=self._detect_language(chunk_text),
                    metadata={"source_type": "pdf"},
                ))

        logger.info(f"Chunked {file_path.name}: {len(chunks)} chunks from {len(reader.pages)} pages")
        return chunks

    def _chunk_docx(self, file_path: Path) -> list[Chunk]:
        """Extract and chunk DOCX preserving heading structure."""
        from docx import Document

        doc = Document(file_path)
        chunks = []
        current_section = ""
        current_text = ""
        page_estimate = 1
        para_idx = 0

        for para in doc.paragraphs:
            if para.style.name.startswith("Heading"):
                # Flush current buffer
                if current_text.strip():
                    for chunk_text in self._split_text(current_text):
                        para_idx += 1
                        chunks.append(Chunk(
                            text=f"[Section: {current_section}]\n{chunk_text}" if current_section else chunk_text,
                            document=file_path.name,
                            page=page_estimate,
                            paragraph=para_idx,
                            language=self._detect_language(chunk_text),
                            metadata={"source_type": "docx", "section": current_section},
                        ))
                current_section = para.text
                current_text = ""
            else:
                current_text += para.text + "\n"
                # Rough page estimation
                if len(current_text) > 3000:
                    page_estimate += 1

        # Flush remaining
        if current_text.strip():
            for chunk_text in self._split_text(current_text):
                para_idx += 1
                chunks.append(Chunk(
                    text=f"[Section: {current_section}]\n{chunk_text}" if current_section else chunk_text,
                    document=file_path.name,
                    page=page_estimate,
                    paragraph=para_idx,
                    language=self._detect_language(chunk_text),
                    metadata={"source_type": "docx", "section": current_section},
                ))

        logger.info(f"Chunked {file_path.name}: {len(chunks)} chunks")
        return chunks

    def _chunk_excel(self, file_path: Path) -> list[Chunk]:
        """Convert Excel spec sheets to searchable text chunks."""
        import pandas as pd

        chunks = []
        xls = pd.ExcelFile(file_path)

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # Convert each row group to text
            text = f"Sheet: {sheet_name}\n"
            text += df.to_string(index=False)

            for idx, chunk_text in enumerate(self._split_text(text)):
                chunks.append(Chunk(
                    text=chunk_text,
                    document=file_path.name,
                    page=1,
                    paragraph=idx + 1,
                    language=self._detect_language(chunk_text),
                    metadata={"source_type": "excel", "sheet": sheet_name},
                ))

        logger.info(f"Chunked {file_path.name}: {len(chunks)} chunks from {len(xls.sheet_names)} sheets")
        return chunks

    def _chunk_text(self, file_path: Path) -> list[Chunk]:
        """Chunk plain text or markdown files."""
        text = file_path.read_text(encoding="utf-8")
        chunks = []
        for idx, chunk_text in enumerate(self._split_text(text)):
            chunks.append(Chunk(
                text=chunk_text,
                document=file_path.name,
                page=1,
                paragraph=idx + 1,
                language=self._detect_language(chunk_text),
                metadata={"source_type": "text"},
            ))
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks respecting sentence boundaries."""
        sentences = self._sentence_split(text)
        chunks = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_length + len(sentence) > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)

                # Keep overlap
                overlap_length = 0
                overlap_start = len(current_chunk)
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_length += len(current_chunk[i])
                    if overlap_length >= self.chunk_overlap:
                        overlap_start = i
                        break
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        """Simple sentence splitter that handles Greek and English."""
        import re
        # Split on sentence-ending punctuation followed by space or newline
        sentences = re.split(r'(?<=[.!?;·])\s+', text)
        return sentences

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect if text is primarily Greek or English."""
        greek_chars = sum(1 for c in text if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        return "el" if greek_chars > latin_chars else "en"


class Indexer:
    """
    Build and persist FAISS + BM25 indices from document chunks.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self.chunker = DocumentChunker()

    def build_index(
        self,
        docs_dir: str | Path,
        output_dir: str | Path,
        glob_pattern: str = "**/*",
    ) -> dict:
        """
        Index all documents in a directory.

        Returns:
            Statistics dict with counts and timing.
        """
        import faiss
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi
        import time

        docs_dir = Path(docs_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all chunks
        t0 = time.perf_counter()
        all_chunks: list[dict] = []
        supported = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md"}
        files = [f for f in docs_dir.glob(glob_pattern) if f.suffix.lower() in supported]

        logger.info(f"Found {len(files)} documents to index")

        for file_path in files:
            chunks = self.chunker.chunk_document(file_path)
            all_chunks.extend(c.to_dict() for c in chunks)

        if not all_chunks:
            logger.warning("No chunks generated — check your documents directory")
            return {"documents": 0, "chunks": 0}

        # Build dense index
        logger.info(f"Embedding {len(all_chunks)} chunks...")
        embedder = SentenceTransformer(self.model_name, device=self.device)
        texts = [c["chunk_text"] for c in all_chunks]
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim with normalized vecs)
        index.add(embeddings.astype(np.float32))

        faiss.write_index(index, str(output_dir / "dense.index"))
        logger.info(f"Dense index saved: {index.ntotal} vectors, dim={dimension}")

        # Build sparse index
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        with open(output_dir / "sparse.pkl", "wb") as f:
            pickle.dump(bm25, f)
        logger.info("Sparse (BM25) index saved")

        # Save chunk store
        with open(output_dir / "chunks.pkl", "wb") as f:
            pickle.dump(all_chunks, f)

        elapsed = time.perf_counter() - t0
        stats = {
            "documents": len(files),
            "chunks": len(all_chunks),
            "embedding_dim": dimension,
            "index_time_seconds": round(elapsed, 2),
        }
        logger.info(f"Indexing complete: {stats}")
        return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index manufacturing documents")
    parser.add_argument("--docs-dir", required=True, help="Directory containing documents")
    parser.add_argument("--output-dir", default="./data/index", help="Output directory for indices")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    indexer = Indexer()
    stats = indexer.build_index(args.docs_dir, args.output_dir)
    print(f"\nIndexing complete: {stats}")
