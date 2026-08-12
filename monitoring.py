import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    firebase_admin = None
    credentials = None
    firestore = None


BASE_DIRECTORY = Path(__file__).parent
SQLITE_DATABASE_PATH = BASE_DIRECTORY / "rag_monitoring.db"
SERVICE_ACCOUNT_PATH = BASE_DIRECTORY / "serviceAccountKey.json"

FIRESTORE_COLLECTION = "rag_monitoring_logs"

_firestore_client = None
_firestore_checked = False


def load_firebase_credentials():
    """Load credentials locally or from Streamlit Secrets."""

    if credentials is None:
        return None

    # On your computer, use the local JSON key
    if SERVICE_ACCOUNT_PATH.exists():
        return credentials.Certificate(
            str(SERVICE_ACCOUNT_PATH)
        )

    # On Streamlit Cloud, use protected Secrets
    try:
        import streamlit as st

        secret_json = st.secrets[
            "FIREBASE_SERVICE_ACCOUNT"
        ]

        service_account_information = json.loads(
            secret_json
        )

        return credentials.Certificate(
            service_account_information
        )

    except Exception as error:
        print(
            "Firebase credentials were not found: "
            f"{error}"
        )

        return None


def get_firestore_client():
    """Connect to Firestore when credentials are available."""

    global _firestore_client
    global _firestore_checked

    if _firestore_checked:
        return _firestore_client

    _firestore_checked = True

    if firebase_admin is None:
        print(
            "Firebase Admin SDK is unavailable. "
            "Using SQLite monitoring."
        )

        return None

    firebase_credentials = load_firebase_credentials()

    if firebase_credentials is None:
        print(
            "Firebase credentials are unavailable. "
            "Using SQLite monitoring."
        )

        return None

    try:
        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(
                firebase_credentials
            )

        _firestore_client = firestore.client()

        print(
            "Firestore monitoring connection successful."
        )

    except Exception as error:
        print(
            "Firestore connection failed. "
            f"Using SQLite instead: {error}"
        )

        _firestore_client = None

    return _firestore_client


def get_sqlite_connection():
    """Create a connection to the local SQLite database."""

    return sqlite3.connect(SQLITE_DATABASE_PATH)


def initialize_monitoring_database():
    """Create the SQLite fallback table."""

    with get_sqlite_connection() as connection:
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
    """Save one request in Firestore or SQLite."""

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    monitoring_record = {
        "created_at": created_at,
        "question": question,
        "answer": answer,
        "source_pages": source_pages,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms,
        "status": status,
        "error_message": error_message,
        "feedback": None
    }

    firestore_client = get_firestore_client()

    if firestore_client is not None:
        document_reference = (
            firestore_client
            .collection(FIRESTORE_COLLECTION)
            .document()
        )

        document_reference.set(monitoring_record)

        return document_reference.id

    with get_sqlite_connection() as connection:
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
    """Save helpful or unhelpful feedback."""

    firestore_client = get_firestore_client()

    if (
        firestore_client is not None
        and isinstance(log_id, str)
    ):
        (
            firestore_client
            .collection(FIRESTORE_COLLECTION)
            .document(log_id)
            .update({
                "feedback": feedback
            })
        )

        return

    with get_sqlite_connection() as connection:
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
    """Read monitoring records from Firestore or SQLite."""

    firestore_client = get_firestore_client()

    if firestore_client is not None:
        try:
            documents = (
                firestore_client
                .collection(FIRESTORE_COLLECTION)
                .order_by(
                    "created_at",
                    direction=firestore.Query.DESCENDING
                )
                .stream()
            )

            records = []

            for document in documents:
                record = document.to_dict()
                record["id"] = document.id

                pages = record.get("source_pages", [])

                if isinstance(pages, list):
                    record["source_pages"] = json.dumps(
                        pages
                    )

                records.append(record)

            return records

        except Exception as error:
            print(
                "Could not read Firestore records. "
                f"Using SQLite instead: {error}"
            )

    with get_sqlite_connection() as connection:
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