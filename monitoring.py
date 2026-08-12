import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


BASE_DIRECTORY = Path(__file__).parent
DATABASE_PATH = BASE_DIRECTORY / "rag_monitoring.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_monitoring_database():
    """Create the monitoring table if it does not already exist."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                source_pages TEXT,
                retrieved_chunks INTEGER DEFAULT 0,
                retrieval_ms REAL DEFAULT 0,
                generation_ms REAL DEFAULT 0,
                total_ms REAL DEFAULT 0,
                status TEXT NOT NULL,
                error_message TEXT,
                feedback TEXT
            )
            """
        )

        connection.commit()


def log_rag_query(
    question,
    answer,
    source_pages,
    retrieved_chunks,
    retrieval_ms,
    generation_ms,
    total_ms,
    status="success",
    error_message=None
):
    """Save one RAG request and return its log ID."""

    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO query_logs (
                created_at,
                question,
                answer,
                source_pages,
                retrieved_chunks,
                retrieval_ms,
                generation_ms,
                total_ms,
                status,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                question,
                answer,
                json.dumps(source_pages),
                retrieved_chunks,
                retrieval_ms,
                generation_ms,
                total_ms,
                status,
                error_message
            )
        )

        connection.commit()
        return cursor.lastrowid


def save_feedback(log_id, feedback):
    """Save helpful or unhelpful feedback for an answer."""

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE query_logs
            SET feedback = ?
            WHERE id = ?
            """,
            (feedback, log_id)
        )

        connection.commit()


def get_monitoring_logs():
    """Return all monitoring records, newest first."""

    with get_connection() as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT *
            FROM query_logs
            ORDER BY id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


initialize_monitoring_database()