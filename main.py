import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import get_session_messages, init_db, list_documents, save_message
from rag_pipeline import RAGService, sources_to_json


load_dotenv()
app = FastAPI(title="Smart Study Assistant")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

rag: RAGService | None = None


def _is_valid_openai_key(key: str | None) -> bool:
    if not key:
        return False
    return key not in {"your_openai_api_key_here", "sk-xxxxxxxxxxxxxxxx"}


@app.on_event("startup")
def startup_event() -> None:
    global rag
    init_db()
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    try:
        rag = RAGService()
    except ValueError:
        rag = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> Any:
    # Use explicit keywords to avoid Starlette signature/version mismatch.
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/health")
def health() -> dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY")
    return {
        "ok": True,
        "openai_configured": _is_valid_openai_key(key),
        "rag_ready": rag is not None,
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    if rag is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing; cannot run RAG.")
    content = await file.read()
    try:
        result = rag.ingest_document(file.filename, content)
        return JSONResponse({"ok": True, "data": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    answer_mode: str = Form("detailed"),
    session_id: str | None = Form(None),
) -> JSONResponse:
    if rag is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing; cannot run RAG.")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    active_session = session_id or str(uuid.uuid4())
    try:
        result = rag.answer_question(question=question, mode=answer_mode)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Question failed: {str(e)}") from e
    now = datetime.now(timezone.utc).isoformat()

    save_message(active_session, "user", question, None, now)
    save_message(active_session, "assistant", result["answer"], sources_to_json(result["sources"]), now)
    return JSONResponse({"ok": True, "session_id": active_session, "data": result})


@app.post("/summarize")
async def summarize(topic: str = Form(...)) -> JSONResponse:
    if rag is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing; cannot run RAG.")
    try:
        summary = rag.summarize_topic(topic)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Summarize failed: {str(e)}") from e
    return JSONResponse({"ok": True, "data": {"summary": summary}})


@app.post("/quiz")
async def quiz(topic: str = Form(...), count: int = Form(5)) -> JSONResponse:
    if rag is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing; cannot run RAG.")
    try:
        quiz_text = rag.generate_quiz(topic, count=count)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Quiz failed: {str(e)}") from e
    return JSONResponse({"ok": True, "data": {"quiz": quiz_text}})


@app.get("/documents")
def documents() -> JSONResponse:
    return JSONResponse({"ok": True, "data": list_documents()})


@app.get("/history/{session_id}")
def history(session_id: str) -> JSONResponse:
    messages = get_session_messages(session_id)
    for msg in messages:
        if msg.get("sources_json"):
            msg["sources"] = json.loads(msg["sources_json"])
        else:
            msg["sources"] = []
    return JSONResponse({"ok": True, "data": messages})
