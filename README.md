# Smart Study Assistant (LLM + RAG)

A web app where students upload their own notes (PDF, DOCX, TXT), ask questions, and get answers grounded in their material with source references.

## Features
- Upload and index `PDF`, `DOCX`, `TXT`
- Text extraction + chunking + embeddings
- Chroma vector store for retrieval
- RAG question answering using OpenAI
- Student-friendly short/detailed answers
- Topic summary and quiz generation
- Source references (file + page + chunk)
- Chat history stored in SQLite by session

## Tech Stack
- Backend: FastAPI
- Frontend: HTML/CSS/JavaScript
- LLM/Embeddings: OpenAI API
- Vector DB: ChromaDB (persistent local)
- Database: SQLite
- Parsers: `pypdf`, `python-docx`

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure env:
   ```bash
   cp .env.example .env
   ```
   Put your OpenAI key in `.env`.

3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

4. Open:
   - [http://127.0.0.1:8000](http://127.0.0.1:8000)

## API Endpoints
- `POST /upload` - Upload and index one file
- `POST /ask` - Ask question (`question`, optional `answer_mode`, `session_id`)
- `POST /summarize` - Summarize topic (`topic`)
- `POST /quiz` - Generate quiz (`topic`, `count`)
- `GET /documents` - List indexed docs
- `GET /history/{session_id}` - Chat history
- `GET /health` - Service health

## Notes
- The assistant is prompted to answer only from retrieved context.
- If no relevant context is found, it reports that clearly.
- Data is stored locally in `data/`.
