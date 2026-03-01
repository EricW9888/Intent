"""SQLite persistence layer for transcription sessions, utterances, and speaker profiles."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "transcriber.db"


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    conn = _connect(db_path)
    
    # 1. Base tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS folders (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            parent_id TEXT REFERENCES folders(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL,
            ended_at TEXT,
            model TEXT NOT NULL DEFAULT '',
            language TEXT DEFAULT '',
            config_json TEXT DEFAULT '{}',
            transcript TEXT NOT NULL DEFAULT '',
            compressed_memory TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active'
        );

        CREATE TABLE IF NOT EXISTS utterances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            idx INTEGER NOT NULL,
            start_seconds REAL NOT NULL,
            end_seconds REAL NOT NULL,
            text TEXT NOT NULL,
            raw_text TEXT NOT NULL DEFAULT '',
            speaker TEXT NOT NULL DEFAULT '',
            forced_split INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_utterances_session ON utterances(session_id);

        CREATE TABLE IF NOT EXISTS speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL UNIQUE,
            embedding_json TEXT NOT NULL,
            sample_count INTEGER NOT NULL DEFAULT 1
        );


        CREATE TABLE IF NOT EXISTS compression_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            utterance_index INTEGER NOT NULL,
            archived_chars INTEGER NOT NULL,
            kept_recent_chars INTEGER NOT NULL,
            compressed_memory_chars INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_compression_session ON compression_events(session_id);

        CREATE TABLE IF NOT EXISTS concept_maps (
            id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
            folder_id TEXT REFERENCES folders(id) ON DELETE CASCADE,
            graph_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_concept_maps_session ON concept_maps(session_id);
        CREATE INDEX IF NOT EXISTS idx_concept_maps_folder ON concept_maps(folder_id);
    """)
    conn.commit()

    # 2. Add folder_id to sessions if missing (schema migration)
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN folder_id TEXT REFERENCES folders(id) ON DELETE SET NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.close()


def create_folder(name: str, parent_id: str | None = None, db_path: Path = DB_PATH) -> str:
    folder_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    conn.execute(
        "INSERT INTO folders (id, name, parent_id, created_at) VALUES (?, ?, ?, ?)",
        (folder_id, name, parent_id, now),
    )
    conn.commit()
    conn.close()
    return folder_id


def rename_folder(folder_id: str, new_name: str, db_path: Path = DB_PATH) -> None:
    conn = _connect(db_path)
    conn.execute("UPDATE folders SET name = ? WHERE id = ?", (new_name, folder_id))
    conn.commit()
    conn.close()


def delete_folder(folder_id: str, db_path: Path = DB_PATH) -> bool:
    conn = _connect(db_path)
    conn.execute("UPDATE sessions SET folder_id = NULL WHERE folder_id = ?", (folder_id,))
    cursor = conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def list_folders(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute("SELECT * FROM folders ORDER BY name ASC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def move_session_to_folder(session_id: str, folder_id: str | None, db_path: Path = DB_PATH) -> None:
    conn = _connect(db_path)
    conn.execute("UPDATE sessions SET folder_id = ? WHERE id = ?", (folder_id, session_id))
    conn.commit()
    conn.close()


def create_session(
    model: str = "",
    language: str = "",
    config: dict[str, Any] | None = None,
    folder_id: str | None = None,
    db_path: Path = DB_PATH,
) -> str:
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    conn.execute(
        "INSERT INTO sessions (id, started_at, model, language, config_json, folder_id) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, now, model, language or "", json.dumps(config or {}), folder_id),
    )
    conn.commit()
    conn.close()
    return session_id



def save_utterance(
    session_id: str,
    index: int,
    start_seconds: float,
    end_seconds: float,
    text: str,
    raw_text: str = "",
    speaker: str = "",
    forced_split: bool = False,
    db_path: Path = DB_PATH,
) -> None:
    conn = _connect(db_path)
    conn.execute(
        """INSERT INTO utterances (session_id, idx, start_seconds, end_seconds, text, raw_text, speaker, forced_split)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, index, round(start_seconds, 3), round(end_seconds, 3), text, raw_text, speaker, int(forced_split)),
    )
    conn.commit()
    conn.close()


def mark_utterance_split(session_id: str, index: int, db_path: Path = DB_PATH) -> None:
    conn = _connect(db_path)
    conn.execute("UPDATE utterances SET forced_split = 1 WHERE session_id = ? AND idx = ?", (session_id, index))
    conn.commit()
    conn.close()


def assign_session_to_folder(session_id: str, folder_id: str | None, db_path: Path = DB_PATH) -> None:
    conn = _connect(db_path)
    conn.execute("UPDATE sessions SET folder_id = ? WHERE id = ?", (folder_id, session_id))
    conn.commit()
    conn.close()


def save_compression_event(
    session_id: str,
    utterance_index: int,
    archived_chars: int,
    kept_recent_chars: int,
    compressed_memory_chars: int,
    db_path: Path = DB_PATH,
) -> None:
    conn = _connect(db_path)
    conn.execute(
        """INSERT INTO compression_events (session_id, utterance_index, archived_chars, kept_recent_chars, compressed_memory_chars)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, utterance_index, archived_chars, kept_recent_chars, compressed_memory_chars),
    )
    conn.commit()
    conn.close()


def update_session_transcript(
    session_id: str,
    transcript: str,
    compressed_memory: str = "",
    db_path: Path = DB_PATH,
) -> None:
    conn = _connect(db_path)
    conn.execute(
        "UPDATE sessions SET transcript = ?, compressed_memory = ? WHERE id = ?",
        (transcript, compressed_memory, session_id),
    )
    conn.commit()
    conn.close()


def end_session(session_id: str, transcript: str = "", compressed_memory: str = "", db_path: Path = DB_PATH) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    conn.execute(
        "UPDATE sessions SET ended_at = ?, status = 'completed', transcript = ?, compressed_memory = ? WHERE id = ?",
        (now, transcript, compressed_memory, session_id),
    )
    conn.commit()
    conn.close()


def list_sessions(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT s.id, s.title, s.started_at, s.ended_at, s.model, s.language, s.status, s.folder_id,
                  COUNT(u.id) as utterance_count,
                  COALESCE(MAX(u.end_seconds), 0) as duration_seconds
           FROM sessions s LEFT JOIN utterances u ON s.id = u.session_id
           GROUP BY s.id ORDER BY s.started_at DESC"""
    ).fetchall()
    conn.close()
    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "title": row["title"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "model": row["model"],
            "language": row["language"],
            "status": row["status"],
            "folder_id": row["folder_id"],
            "utterance_count": row["utterance_count"],
            "duration_seconds": round(row["duration_seconds"], 1),
        })
    return results


def get_session(session_id: str, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        conn.close()
        return None
    utterances = conn.execute(
        "SELECT * FROM utterances WHERE session_id = ? ORDER BY idx", (session_id,)
    ).fetchall()
    compressions = conn.execute(
        "SELECT * FROM compression_events WHERE session_id = ? ORDER BY utterance_index", (session_id,)
    ).fetchall()
    conn.close()
    return {
        "id": row["id"],
        "title": row["title"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "model": row["model"],
        "language": row["language"],
        "status": row["status"],
        "folder_id": row["folder_id"],
        "transcript": row["transcript"],
        "compressed_memory": row["compressed_memory"],
        "config": json.loads(row["config_json"]) if row["config_json"] else {},
        "utterances": [
            {
                "index": u["idx"],
                "start_seconds": u["start_seconds"],
                "end_seconds": u["end_seconds"],
                "text": u["text"],
                "raw_text": u["raw_text"],
                "speaker": u["speaker"],
                "forced_split": bool(u["forced_split"]),
            }
            for u in utterances
        ],
        "compression_events": [
            {
                "utterance_index": c["utterance_index"],
                "archived_chars": c["archived_chars"],
                "kept_recent_chars": c["kept_recent_chars"],
                "compressed_memory_chars": c["compressed_memory_chars"],
            }
            for c in compressions
        ],
    }


def delete_session(session_id: str, db_path: Path = DB_PATH) -> bool:
    conn = _connect(db_path)
    conn.execute("DELETE FROM utterances WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM compression_events WHERE session_id = ?", (session_id,))
    cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def update_session_title(session_id: str, title: str, db_path: Path = DB_PATH) -> None:
    conn = _connect(db_path)
    conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
    conn.commit()
    conn.close()


def save_concept_map(
    graph_json: str,
    session_id: str | None = None,
    folder_id: str | None = None,
    db_path: Path = DB_PATH,
) -> str:
    """Save or replace a concept map for a session or folder."""
    conn = _connect(db_path)
    # Delete existing map for this session/folder
    if session_id:
        conn.execute("DELETE FROM concept_maps WHERE session_id = ?", (session_id,))
    elif folder_id:
        conn.execute("DELETE FROM concept_maps WHERE folder_id = ? AND session_id IS NULL", (folder_id,))
    map_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO concept_maps (id, session_id, folder_id, graph_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (map_id, session_id, folder_id, graph_json, now),
    )
    conn.commit()
    conn.close()
    return map_id


def get_concept_map(session_id: str | None = None, folder_id: str | None = None, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    """Get a concept map by session or folder."""
    conn = _connect(db_path)
    if session_id:
        row = conn.execute("SELECT * FROM concept_maps WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,)).fetchone()
    elif folder_id:
        row = conn.execute("SELECT * FROM concept_maps WHERE folder_id = ? AND session_id IS NULL ORDER BY created_at DESC LIMIT 1", (folder_id,)).fetchone()
    else:
        conn.close()
        return None
    conn.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "folder_id": row["folder_id"],
        "graph": json.loads(row["graph_json"]),
        "created_at": row["created_at"],
    }


def get_folder_session_transcripts(folder_id: str | None, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Get all session transcripts in a folder for building folder-level maps."""
    conn = _connect(db_path)
    if folder_id is None:
        rows = conn.execute(
            "SELECT id, title, transcript FROM sessions WHERE folder_id IS NULL ORDER BY started_at"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, title, transcript FROM sessions WHERE folder_id = ? ORDER BY started_at",
            (folder_id,),
        ).fetchall()
    results = []
    for r in rows:
        transcript = r["transcript"] or ""
        if not transcript.strip():
            # Build from utterances as fallback
            utts = conn.execute(
                "SELECT speaker, text FROM utterances WHERE session_id = ? ORDER BY idx",
                (r["id"],),
            ).fetchall()
            transcript = "\n".join(
                f"[{u['speaker']}] {u['text']}" for u in utts if u["text"].strip()
            )
        if transcript.strip():
            results.append({"id": r["id"], "title": r["title"], "transcript": transcript})
    conn.close()
    return results
