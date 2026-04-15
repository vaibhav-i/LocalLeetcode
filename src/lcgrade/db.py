from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1

_SCHEMA_V1 = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS problems (
    slug TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    category TEXT NOT NULL DEFAULT '',
    function_name TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '[]',
    return_type TEXT NOT NULL DEFAULT '',
    validator TEXT NOT NULL DEFAULT 'exact_match',
    schema_version INTEGER NOT NULL DEFAULT 1,
    file_path TEXT NOT NULL,
    statement_hash TEXT NOT NULL DEFAULT '',
    statement_mtime REAL NOT NULL DEFAULT 0,
    tests_hash TEXT,
    tests_mtime REAL,
    scaling_inputs TEXT,
    auto_solved INTEGER NOT NULL DEFAULT 0,
    auto_solved_at TEXT,
    manual_solved INTEGER NOT NULL DEFAULT 0,
    manual_solved_at TEXT,
    review_generated INTEGER NOT NULL DEFAULT 0,
    review_generated_at TEXT,
    review_acknowledged INTEGER NOT NULL DEFAULT 0,
    review_acknowledged_at TEXT,
    followup_completed INTEGER NOT NULL DEFAULT 0,
    followup_completed_at TEXT,
    CHECK (auto_solved IN (0, 1)),
    CHECK (manual_solved IN (0, 1)),
    CHECK (review_generated IN (0, 1)),
    CHECK (review_acknowledged IN (0, 1)),
    CHECK (followup_completed IN (0, 1))
);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    test_mode TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    tests_hash TEXT NOT NULL,
    bundled_passed INTEGER NOT NULL DEFAULT 0,
    bundled_total INTEGER NOT NULL DEFAULT 0,
    llm_passed INTEGER NOT NULL DEFAULT 0,
    llm_total INTEGER NOT NULL DEFAULT 0,
    runtime_ms REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    code_snapshot TEXT NOT NULL,
    FOREIGN KEY (slug) REFERENCES problems(slug) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_attempts_cache_key
    ON attempts(slug, code_hash, test_mode, tests_hash, id DESC);

CREATE INDEX IF NOT EXISTS idx_attempts_slug_timestamp
    ON attempts(slug, timestamp DESC);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER UNIQUE,
    timestamp TEXT NOT NULL,
    complexity_time TEXT NOT NULL DEFAULT '',
    complexity_space TEXT NOT NULL DEFAULT '',
    review_text TEXT NOT NULL,
    FOREIGN KEY (attempt_id) REFERENCES attempts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reviews_attempt_id
    ON reviews(attempt_id);

CREATE TABLE IF NOT EXISTS extension_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL,
    extension_name TEXT NOT NULL,
    output_text TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_extension_results_review_id
    ON extension_results(review_id);

CREATE INDEX IF NOT EXISTS idx_extension_results_name
    ON extension_results(extension_name);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    message TEXT NOT NULL,
    hint_tier INTEGER,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (slug) REFERENCES problems(slug) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(slug, session_id, timestamp);
"""

MIGRATIONS: dict[int, str] = {
    1: _SCHEMA_V1,
}


@dataclass(slots=True, frozen=True)
class CacheKey:
    code_hash: str
    test_mode: str
    tests_hash: str


def configure_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def open_database(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    return configure_connection(conn)


def get_metadata_value(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return default
    if row is None:
        return default
    return str(row["value"])


def set_metadata_value(conn: sqlite3.Connection, key: str, value: str | None) -> None:
    with conn:
        if value is None:
            conn.execute("DELETE FROM metadata WHERE key = ?", (key,))
            return
        conn.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def get_db_version(conn: sqlite3.Connection) -> int:
    value = get_metadata_value(conn, "db_version", "0")
    return int(value or 0)


def set_db_version(conn: sqlite3.Connection, version: int) -> None:
    set_metadata_value(conn, "db_version", str(version))


def get_active_slug(conn: sqlite3.Connection) -> str | None:
    return get_metadata_value(conn, "active_slug")


def set_active_slug(conn: sqlite3.Connection, slug: str | None) -> None:
    set_metadata_value(conn, "active_slug", slug)


def clear_active_slug(conn: sqlite3.Connection) -> None:
    set_active_slug(conn, None)


def migrate(conn: sqlite3.Connection, target_version: int = SCHEMA_VERSION) -> None:
    current_version = get_db_version(conn)
    if current_version > target_version:
        raise RuntimeError(
            f"Database version {current_version} is newer than supported version {target_version}"
        )

    for version in range(current_version + 1, target_version + 1):
        script = MIGRATIONS.get(version)
        if script is None:
            raise RuntimeError(f"No migration script registered for schema version {version}")
        with conn:
            conn.executescript(script)
            set_db_version(conn, version)


def bootstrap_database(path: str | Path, target_version: int = SCHEMA_VERSION) -> sqlite3.Connection:
    conn = open_database(path)
    migrate(conn, target_version=target_version)
    return conn


def ensure_metadata_table(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def fetch_problem_record(conn: sqlite3.Connection, slug: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM problems WHERE slug = ?", (slug,)).fetchone()
    if row is None:
        return None
    return dict(row)


def list_problem_records(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM problems ORDER BY slug").fetchall()
    return [dict(row) for row in rows]


def upsert_problem_record(conn: sqlite3.Connection, record: Mapping[str, Any]) -> None:
    columns = (
        "slug",
        "title",
        "difficulty",
        "tags",
        "category",
        "function_name",
        "params",
        "return_type",
        "validator",
        "schema_version",
        "file_path",
        "statement_hash",
        "statement_mtime",
        "tests_hash",
        "tests_mtime",
        "scaling_inputs",
    )
    values = tuple(record.get(column) for column in columns)
    with conn:
        conn.execute(
            f"""
            INSERT INTO problems ({", ".join(columns)})
            VALUES ({", ".join(["?"] * len(columns))})
            ON CONFLICT(slug) DO UPDATE SET
                title = excluded.title,
                difficulty = excluded.difficulty,
                tags = excluded.tags,
                category = excluded.category,
                function_name = excluded.function_name,
                params = excluded.params,
                return_type = excluded.return_type,
                validator = excluded.validator,
                schema_version = excluded.schema_version,
                file_path = excluded.file_path,
                statement_hash = excluded.statement_hash,
                statement_mtime = excluded.statement_mtime,
                tests_hash = excluded.tests_hash,
                tests_mtime = excluded.tests_mtime,
                scaling_inputs = excluded.scaling_inputs
            """,
            values,
        )


def delete_problem_records(conn: sqlite3.Connection, slugs: Iterable[str]) -> int:
    slug_list = list(slugs)
    if not slug_list:
        return 0
    with conn:
        placeholders = ", ".join(["?"] * len(slug_list))
        cursor = conn.execute(f"DELETE FROM problems WHERE slug IN ({placeholders})", slug_list)
    return cursor.rowcount


def insert_attempt(
    conn: sqlite3.Connection,
    *,
    slug: str,
    test_mode: str,
    code_hash: str,
    tests_hash: str,
    bundled_passed: int,
    bundled_total: int,
    llm_passed: int,
    llm_total: int,
    runtime_ms: float,
    status: str,
    code_snapshot: str,
) -> int:
    timestamp = datetime.now(timezone.utc).isoformat()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO attempts (
                slug,
                timestamp,
                test_mode,
                code_hash,
                tests_hash,
                bundled_passed,
                bundled_total,
                llm_passed,
                llm_total,
                runtime_ms,
                status,
                code_snapshot
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                slug,
                timestamp,
                test_mode,
                code_hash,
                tests_hash,
                bundled_passed,
                bundled_total,
                llm_passed,
                llm_total,
                runtime_ms,
                status,
                code_snapshot,
            ),
        )
    return int(cursor.lastrowid)


def find_cached_attempt(conn: sqlite3.Connection, slug: str, cache_key: CacheKey) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT *
        FROM attempts
        WHERE slug = ? AND code_hash = ? AND test_mode = ? AND tests_hash = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (slug, cache_key.code_hash, cache_key.test_mode, cache_key.tests_hash),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def set_problem_milestone(
    conn: sqlite3.Connection,
    *,
    slug: str,
    flag_column: str,
    timestamp_column: str,
) -> bool:
    """Set a sticky problem milestone if it has not already been set.

    Returns True when the milestone transitioned from false to true.
    """

    allowed_pairs = {
        ("auto_solved", "auto_solved_at"),
        ("manual_solved", "manual_solved_at"),
        ("review_generated", "review_generated_at"),
        ("review_acknowledged", "review_acknowledged_at"),
        ("followup_completed", "followup_completed_at"),
    }
    if (flag_column, timestamp_column) not in allowed_pairs:
        raise ValueError(f"Unsupported milestone columns: {flag_column}, {timestamp_column}")

    timestamp = datetime.now(timezone.utc).isoformat()
    with conn:
        cursor = conn.execute(
            f"""
            UPDATE problems
            SET
                {flag_column} = 1,
                {timestamp_column} = COALESCE({timestamp_column}, ?)
            WHERE slug = ? AND {flag_column} = 0
            """,
            (timestamp, slug),
        )
    return cursor.rowcount > 0


def mark_problem_auto_solved(conn: sqlite3.Connection, slug: str) -> bool:
    return set_problem_milestone(
        conn,
        slug=slug,
        flag_column="auto_solved",
        timestamp_column="auto_solved_at",
    )


def mark_problem_review_generated(conn: sqlite3.Connection, slug: str) -> bool:
    return set_problem_milestone(
        conn,
        slug=slug,
        flag_column="review_generated",
        timestamp_column="review_generated_at",
    )


def reset_problem_milestones(conn: sqlite3.Connection, slug: str) -> bool:
    with conn:
        cursor = conn.execute(
            """
            UPDATE problems
            SET
                auto_solved = 0,
                auto_solved_at = NULL,
                manual_solved = 0,
                manual_solved_at = NULL,
                review_generated = 0,
                review_generated_at = NULL,
                review_acknowledged = 0,
                review_acknowledged_at = NULL,
                followup_completed = 0,
                followup_completed_at = NULL
            WHERE slug = ?
            """,
            (slug,),
        )
    return cursor.rowcount > 0


def reset_problem_state(conn: sqlite3.Connection, slug: str) -> dict[str, int | bool]:
    with conn:
        chat_cursor = conn.execute("DELETE FROM chat_messages WHERE slug = ?", (slug,))
        attempt_cursor = conn.execute("DELETE FROM attempts WHERE slug = ?", (slug,))
    milestones_reset = reset_problem_milestones(conn, slug)
    cleared_active_slug = False
    if get_active_slug(conn) == slug:
        clear_active_slug(conn)
        cleared_active_slug = True
    return {
        "attempts_deleted": int(attempt_cursor.rowcount),
        "chat_messages_deleted": int(chat_cursor.rowcount),
        "milestones_reset": milestones_reset,
        "cleared_active_slug": cleared_active_slug,
    }


def clear_chat_messages(conn: sqlite3.Connection, slug: str) -> int:
    with conn:
        cursor = conn.execute("DELETE FROM chat_messages WHERE slug = ?", (slug,))
    return int(cursor.rowcount)


def prune_attempt_history(conn: sqlite3.Connection, *, keep_per_problem: int = 10) -> int:
    if keep_per_problem < 0:
        raise ValueError("keep_per_problem must be non-negative")

    deleted = 0
    rows = conn.execute("SELECT slug FROM problems ORDER BY slug").fetchall()
    for row in rows:
        slug = str(row["slug"])
        attempt_rows = conn.execute(
            """
            SELECT id
            FROM attempts
            WHERE slug = ?
            ORDER BY id DESC
            """,
            (slug,),
        ).fetchall()
        ids_to_delete = [int(item["id"]) for item in attempt_rows[keep_per_problem:]]
        if not ids_to_delete:
            continue
        placeholders = ", ".join(["?"] * len(ids_to_delete))
        with conn:
            cursor = conn.execute(
                f"DELETE FROM attempts WHERE id IN ({placeholders})",
                ids_to_delete,
            )
        deleted += int(cursor.rowcount)
    return deleted


def insert_chat_message(
    conn: sqlite3.Connection,
    *,
    slug: str,
    session_id: str,
    role: str,
    message: str,
    hint_tier: int | None = None,
) -> int:
    timestamp = datetime.now(timezone.utc).isoformat()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO chat_messages (
                slug,
                session_id,
                role,
                message,
                hint_tier,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (slug, session_id, role, message, hint_tier, timestamp),
        )
    return int(cursor.lastrowid)


def fetch_recent_chat_messages(
    conn: sqlite3.Connection,
    *,
    slug: str,
    session_id: str = "default",
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM chat_messages
        WHERE slug = ? AND session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (slug, session_id, limit),
    ).fetchall()
    return [dict(row) for row in reversed(rows)]
