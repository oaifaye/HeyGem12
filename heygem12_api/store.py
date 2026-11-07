"""SQLite persistence for assets, lipsync jobs, batches and idempotency keys."""
import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY, filename TEXT, path TEXT NOT NULL, mime_type TEXT,
    size_bytes INTEGER, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS lipsyncs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL, status TEXT NOT NULL, title TEXT,
    request TEXT NOT NULL,            -- normalized request JSON
    callback_url TEXT, callback_id TEXT, base_url TEXT,
    batch_id TEXT, item_index INTEGER,
    created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER,
    duration REAL, output_path TEXT, failure_code TEXT, failure_message TEXT);
CREATE INDEX IF NOT EXISTS ix_lipsyncs_status ON lipsyncs(status, seq);
CREATE INDEX IF NOT EXISTS ix_lipsyncs_batch ON lipsyncs(batch_id, item_index);
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY, title TEXT, callback_url TEXT, base_url TEXT,
    total_items INTEGER NOT NULL, created_at INTEGER NOT NULL, notified INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT NOT NULL, scope TEXT NOT NULL, status_code INTEGER, response TEXT,
    created_at INTEGER NOT NULL, PRIMARY KEY (key, scope));
"""

TERMINAL = ("completed", "failed")


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> int:
    return int(time.time())


class Store:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def _q(self, sql: str, args=()) -> List[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _x(self, sql: str, args=()) -> int:
        with self._lock:
            return self._db.execute(sql, args).rowcount

    # ---------------------------------------------------------------- assets
    def add_asset(self, asset_id, filename, path, mime, size) -> Dict[str, Any]:
        self._x("INSERT INTO assets VALUES (?,?,?,?,?,?)", (asset_id, filename, path, mime, size, now()))
        return self.get_asset(asset_id)

    def get_asset(self, asset_id) -> Optional[Dict[str, Any]]:
        r = self._q("SELECT * FROM assets WHERE id=?", (asset_id,))
        return dict(r[0]) if r else None

    def delete_asset(self, asset_id) -> bool:
        return self._x("DELETE FROM assets WHERE id=?", (asset_id,)) > 0

    # -------------------------------------------------------------- lipsyncs
    def add_lipsync(self, req: Dict[str, Any], base_url: str, batch_id=None, item_index=None) -> str:
        lid = new_id()
        self._x("""INSERT INTO lipsyncs (id, status, title, request, callback_url, callback_id,
                   base_url, batch_id, item_index, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (lid, "pending", req.get("title"), json.dumps(req), req.get("callback_url"),
                 req.get("callback_id"), base_url, batch_id, item_index, now()))
        return lid

    def get_lipsync(self, lid) -> Optional[Dict[str, Any]]:
        r = self._q("SELECT * FROM lipsyncs WHERE id=?", (lid,))
        return dict(r[0]) if r else None

    def list_lipsyncs(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._q(
            "SELECT * FROM lipsyncs ORDER BY seq DESC LIMIT ? OFFSET ?", (limit + 1, offset))]

    def batch_items(self, batch_id, limit: int, offset: int) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._q(
            "SELECT * FROM lipsyncs WHERE batch_id=? ORDER BY item_index LIMIT ? OFFSET ?",
            (batch_id, limit + 1, offset))]

    def batch_counts(self, batch_id) -> Dict[str, int]:
        return {r["status"]: r["n"] for r in self._q(
            "SELECT status, COUNT(*) n FROM lipsyncs WHERE batch_id=? GROUP BY status", (batch_id,))}

    def update_lipsync(self, lid, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE lipsyncs SET {cols} WHERE id=?", (*fields.values(), lid))

    def delete_lipsync(self, lid) -> bool:
        return self._x("DELETE FROM lipsyncs WHERE id=?", (lid,)) > 0

    def claim_next(self) -> Optional[Dict[str, Any]]:
        """Atomically move the oldest pending job to running."""
        with self._lock:
            r = self._q("SELECT id FROM lipsyncs WHERE status='pending' ORDER BY seq LIMIT 1")
            if not r:
                return None
            self._x("UPDATE lipsyncs SET status='running', started_at=? WHERE id=? AND status='pending'",
                    (now(), r[0]["id"]))
            return self.get_lipsync(r[0]["id"])

    def queue_length(self) -> int:
        return self._q("SELECT COUNT(*) n FROM lipsyncs WHERE status='pending'")[0]["n"]

    def requeue_running(self) -> int:
        """At startup, jobs left 'running' by a crash/restart go back to the queue."""
        return self._x("UPDATE lipsyncs SET status='pending', started_at=NULL WHERE status='running'")

    # --------------------------------------------------------------- batches
    def add_batch(self, title, callback_url, base_url, total) -> str:
        bid = new_id()
        self._x("INSERT INTO batches (id, title, callback_url, base_url, total_items, created_at) "
                "VALUES (?,?,?,?,?,?)", (bid, title, callback_url, base_url, total, now()))
        return bid

    def get_batch(self, bid) -> Optional[Dict[str, Any]]:
        r = self._q("SELECT * FROM batches WHERE id=?", (bid,))
        return dict(r[0]) if r else None

    def mark_batch_notified(self, bid) -> bool:
        """True exactly once per batch (guards against duplicate batch webhooks)."""
        return self._x("UPDATE batches SET notified=1 WHERE id=? AND notified=0", (bid,)) > 0

    # ----------------------------------------------------------- idempotency
    def idem_get(self, key, scope, ttl=24 * 3600):
        r = self._q("SELECT * FROM idempotency WHERE key=? AND scope=? AND created_at>?",
                    (key, scope, now() - ttl))
        return dict(r[0]) if r else None

    def idem_reserve(self, key, scope) -> bool:
        """Reserve a key (status_code NULL = in flight). False if already present and live."""
        with self._lock:
            self._x("DELETE FROM idempotency WHERE created_at<?", (now() - 24 * 3600,))
            try:
                self._x("INSERT INTO idempotency (key, scope, created_at) VALUES (?,?,?)", (key, scope, now()))
                return True
            except sqlite3.IntegrityError:
                return False

    def idem_store(self, key, scope, status_code, response) -> None:
        self._x("UPDATE idempotency SET status_code=?, response=? WHERE key=? AND scope=?",
                (status_code, json.dumps(response), key, scope))

    def idem_release(self, key, scope) -> None:
        self._x("DELETE FROM idempotency WHERE key=? AND scope=? AND status_code IS NULL", (key, scope))
