"""Regression tests for the strictly read-only state.db projection."""

import sqlite3
from pathlib import Path

import pytest

from api.agent_sessions import open_state_db_readonly, read_importable_agent_session_rows


def _make_state_db(path: Path, *, with_index: bool = False) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            model TEXT,
            message_count INTEGER,
            started_at REAL,
            ended_at REAL,
            parent_session_id TEXT,
            archived INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            timestamp REAL,
            role TEXT,
            content TEXT
        );
        INSERT INTO sessions VALUES
            ('cron-1', 'cron', 'Nightly job', 'test-model', 1, 10, 20, NULL, 0);
        INSERT INTO messages VALUES
            (1, 'cron-1', 20, 'user', 'run');
        """
    )
    if with_index:
        conn.execute(
            "CREATE INDEX idx_messages_session ON messages(session_id, timestamp)"
        )
    conn.commit()
    conn.close()


def test_open_state_db_readonly_cannot_write(tmp_path):
    db_path = tmp_path / "state.db"
    _make_state_db(db_path)

    conn = open_state_db_readonly(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO sessions(id, source) VALUES ('x', 'cli')")
    finally:
        conn.close()

    check = sqlite3.connect(db_path)
    try:
        assert check.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    finally:
        check.close()


def test_session_projection_does_not_create_missing_index(tmp_path):
    db_path = tmp_path / "state.db"
    _make_state_db(db_path)

    rows = read_importable_agent_session_rows(
        db_path, limit=20, include_sources=("cron",), exclude_sources=None
    )

    assert [row["id"] for row in rows] == ["cron-1"]
    check = sqlite3.connect(db_path)
    try:
        indexes = {
            row[1]
            for row in check.execute("PRAGMA index_list(messages)").fetchall()
        }
        assert "idx_messages_session" not in indexes
    finally:
        check.close()


def test_cache_owned_projection_raises_instead_of_reporting_no_sessions(tmp_path, monkeypatch):
    """A failed read-only open must RAISE for cache-owned callers, not return [].

    The CLI-sessions cache treats a returned list as authoritative and stores it.
    If a transient open failure (EMFILE, locked/damaged DB, exotic FS) returned an
    empty list, the cache would overwrite a known-good populated projection and the
    session sidebar would go blank until the next successful rebuild. Raising routes
    the failure to the stale-cache handler, which returns the previous projection.

    Note this is deliberately NOT a writable fallback: the projection must never
    open state.db for writing (that is the whole point of #7555).
    """
    db_path = tmp_path / "state.db"
    _make_state_db(db_path)

    # Sanity: the projection reads fine before we break the open.
    assert [row["id"] for row in read_importable_agent_session_rows(
        db_path, limit=20, include_sources=("cron",), exclude_sources=None
    )] == ["cron-1"]

    def _boom(*_a, **_kw):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr("api.agent_sessions.open_state_db_readonly", _boom)

    # Default (non-cache) callers still degrade to an empty projection...
    assert read_importable_agent_session_rows(
        db_path, limit=20, include_sources=("cron",), exclude_sources=None
    ) == []

    # ...but a cache-owned caller must see the failure so it can keep stale rows.
    with pytest.raises(sqlite3.Error):
        read_importable_agent_session_rows(
            db_path,
            limit=20,
            include_sources=("cron",),
            exclude_sources=None,
            raise_on_unavailable=True,
        )
