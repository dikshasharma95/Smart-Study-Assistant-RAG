import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


DB_PATH = Path("data/app.db")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                total_chunks INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def get_conn() -> Any:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def save_document(file_name: str, file_type: str, uploaded_at: str, total_chunks: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents(file_name, file_type, uploaded_at, total_chunks) VALUES (?, ?, ?, ?)",
            (file_name, file_type, uploaded_at, total_chunks),
        )
        conn.commit()


def list_documents() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, file_name, file_type, uploaded_at, total_chunks FROM documents ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def save_message(
    session_id: str,
    role: str,
    content: str,
    sources_json: str | None,
    created_at: str,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages(session_id, role, content, sources_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, sources_json, created_at),
        )
        conn.commit()


def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, role, content, sources_json, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]
