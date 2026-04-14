import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from docx import Document
from openai import OpenAI
from pypdf import PdfReader

from database import save_document


UPLOAD_DIR = Path("data/uploads")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "study_chunks"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


class RAGService:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key in {"your_openai_api_key_here", "sk-xxxxxxxxxxxxxxxx"}:
            raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")
        self.openai = OpenAI(api_key=api_key)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def parse_file(self, file_path: Path, file_type: str) -> list[dict[str, Any]]:
        if file_type == ".pdf":
            return self._parse_pdf(file_path)
        if file_type == ".docx":
            return self._parse_docx(file_path)
        if file_type == ".txt":
            return self._parse_txt(file_path)
        raise ValueError("Unsupported file type")

    def _parse_pdf(self, file_path: Path) -> list[dict[str, Any]]:
        reader = PdfReader(str(file_path))
        docs: list[dict[str, Any]] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append({"text": text, "page": i})
        return docs

    def _parse_docx(self, file_path: Path) -> list[dict[str, Any]]:
        document = Document(str(file_path))
        content = "\n".join([p.text for p in document.paragraphs if p.text.strip()])
        return [{"text": content, "page": None}] if content.strip() else []

    def _parse_txt(self, file_path: Path) -> list[dict[str, Any]]:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return [{"text": text, "page": None}] if text.strip() else []

    def chunk_text(self, text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def ingest_document(self, file_name: str, file_bytes: bytes) -> dict[str, Any]:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_path = UPLOAD_DIR / file_name
        file_path.write_bytes(file_bytes)

        ext = file_path.suffix.lower()
        if ext not in {".pdf", ".docx", ".txt"}:
            raise ValueError("Only PDF, DOCX, and TXT are supported.")

        docs = self.parse_file(file_path, ext)
        if not docs:
            raise ValueError("Could not extract useful text from this file.")

        all_texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for unit in docs:
            chunks = self.chunk_text(unit["text"])
            for idx, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                ids.append(chunk_id)
                all_texts.append(chunk)
                metadatas.append(
                    {
                        "file_name": file_name,
                        "page": unit["page"] if unit["page"] is not None else -1,
                        "chunk_index": idx,
                    }
                )

        vectors = self.openai.embeddings.create(model=EMBED_MODEL, input=all_texts)
        embeddings = [item.embedding for item in vectors.data]
        self.collection.add(ids=ids, documents=all_texts, metadatas=metadatas, embeddings=embeddings)

        uploaded_at = datetime.now(timezone.utc).isoformat()
        save_document(file_name=file_name, file_type=ext, uploaded_at=uploaded_at, total_chunks=len(all_texts))
        return {"file_name": file_name, "chunks_indexed": len(all_texts)}

    def retrieve(self, question: str, k: int = 5) -> list[dict[str, Any]]:
        query_vector = self.openai.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
        result = self.collection.query(query_embeddings=[query_vector], n_results=k)

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[dict[str, Any]] = []
        for d, m, dist in zip(docs, metas, distances):
            chunks.append(
                {
                    "text": d,
                    "file_name": m.get("file_name", "unknown"),
                    "page": m.get("page", -1),
                    "chunk_index": m.get("chunk_index", -1),
                    "distance": dist,
                }
            )
        return chunks

    def answer_question(self, question: str, mode: str = "detailed") -> dict[str, Any]:
        chunks = self.retrieve(question)
        if not chunks:
            return {
                "answer": "I could not find relevant information in your uploaded study material.",
                "sources": [],
            }

        context = "\n\n".join(
            [
                f"[Source {i+1}] File: {c['file_name']} | Page: {c['page']} | Chunk: {c['chunk_index']}\n{c['text']}"
                for i, c in enumerate(chunks)
            ]
        )

        style = (
            "Give a short, direct answer in simple student-friendly language."
            if mode == "short"
            else "Give a clear, detailed answer in simple student-friendly language with key points."
        )
        prompt = f"""
You are a Smart Study Assistant.
Use only the provided context. Do not use outside knowledge.
If the answer is not in context, say that clearly.

Style: {style}

Question:
{question}

Context:
{context}
"""
        completion = self.openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = completion.choices[0].message.content or ""
        sources = [
            {
                "file_name": c["file_name"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ]
        return {"answer": answer.strip(), "sources": sources}

    def summarize_topic(self, topic: str) -> str:
        chunks = self.retrieve(topic, k=6)
        if not chunks:
            return "No matching content found in uploaded material."
        context = "\n\n".join([c["text"] for c in chunks])
        prompt = f"Summarize this topic in simple language for students:\nTopic: {topic}\nContext:\n{context}"
        completion = self.openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return (completion.choices[0].message.content or "").strip()

    def generate_quiz(self, topic: str, count: int = 5) -> str:
        chunks = self.retrieve(topic, k=6)
        if not chunks:
            return "No matching content found in uploaded material."
        context = "\n\n".join([c["text"] for c in chunks])
        prompt = (
            f"Create {count} quiz questions (with answers) from this context in student-friendly format.\n"
            f"Topic: {topic}\nContext:\n{context}"
        )
        completion = self.openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}],
        )
        return (completion.choices[0].message.content or "").strip()


def sources_to_json(sources: list[dict[str, Any]]) -> str:
    return json.dumps(sources, ensure_ascii=True)
