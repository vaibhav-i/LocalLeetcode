from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import random
import sqlite3
from typing import Any, Iterable, Mapping

from .logging_utils import get_logger

logger = get_logger("lcgrade.db")

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
    logger.info("Opening database at %s", path)
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
    logger.info("Setting active slug to %s", slug)
    set_metadata_value(conn, "active_slug", slug)


def clear_active_slug(conn: sqlite3.Connection) -> None:
    logger.info("Clearing active slug")
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
    logger.info("Inserted attempt for %s status=%s test_mode=%s", slug, status, test_mode)
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
        logger.info("No cached attempt for %s mode=%s", slug, cache_key.test_mode)
        return None
    logger.info("Found cached attempt for %s mode=%s", slug, cache_key.test_mode)
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
    if cursor.rowcount > 0:
        logger.info("Set milestone %s for %s", flag_column, slug)
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
    logger.info("Resetting problem state for %s", slug)
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
    logger.info("Cleared %s chat message(s) for %s", cursor.rowcount, slug)
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
        logger.info("Pruned %s attempt(s) for %s", cursor.rowcount, slug)
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
    logger.info("Inserted chat message for %s session=%s role=%s", slug, session_id, role)
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


def _decode_json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return decoded
        return [decoded]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _record_get(record: Mapping[str, Any] | sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return record[key]
    except (KeyError, IndexError, TypeError):
        return default


def _problem_tags(record: Mapping[str, Any]) -> tuple[str, ...]:
    tags: list[str] = []
    for tag in _decode_json_list(_record_get(record, "tags", "[]")):
        normalized = str(tag).strip().lower()
        if normalized:
            tags.append(normalized)
    return tuple(tags)


def _difficulty_sort_key(value: str) -> tuple[int, str]:
    normalized = value.strip().lower()
    order = {"easy": 0, "medium": 1, "hard": 2}
    return (order.get(normalized, 99), normalized)


def fetch_global_progress_counts(conn: sqlite3.Connection) -> dict[str, int]:
    total_problems = int(conn.execute("SELECT COUNT(*) AS count FROM problems").fetchone()["count"])
    attempted = int(conn.execute("SELECT COUNT(DISTINCT slug) AS count FROM attempts").fetchone()["count"])
    solved = int(
        conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM problems
            WHERE auto_solved = 1 OR manual_solved = 1
            """
        ).fetchone()["count"]
    )
    auto_solved = int(
        conn.execute("SELECT COUNT(*) AS count FROM problems WHERE auto_solved = 1").fetchone()["count"]
    )
    manual_solved = int(
        conn.execute("SELECT COUNT(*) AS count FROM problems WHERE manual_solved = 1").fetchone()["count"]
    )
    review_generated = int(
        conn.execute("SELECT COUNT(*) AS count FROM problems WHERE review_generated = 1").fetchone()["count"]
    )
    review_acknowledged = int(
        conn.execute(
            "SELECT COUNT(*) AS count FROM problems WHERE review_acknowledged = 1"
        ).fetchone()["count"]
    )
    followup_completed = int(
        conn.execute(
            "SELECT COUNT(*) AS count FROM problems WHERE followup_completed = 1"
        ).fetchone()["count"]
    )
    return {
        "total_problems": total_problems,
        "attempted": attempted,
        "solved": solved,
        "auto_solved": auto_solved,
        "manual_solved": manual_solved,
        "review_generated": review_generated,
        "review_acknowledged": review_acknowledged,
        "followup_completed": followup_completed,
    }


def fetch_progress_counts_by_difficulty(conn: sqlite3.Connection) -> list[dict[str, int | str]]:
    rows = conn.execute(
        """
        SELECT
            difficulty,
            COUNT(*) AS total,
            SUM(CASE WHEN EXISTS (SELECT 1 FROM attempts WHERE attempts.slug = problems.slug) THEN 1 ELSE 0 END) AS attempted,
            SUM(CASE WHEN auto_solved = 1 OR manual_solved = 1 THEN 1 ELSE 0 END) AS solved,
            SUM(CASE WHEN review_generated = 1 THEN 1 ELSE 0 END) AS review_generated,
            SUM(CASE WHEN review_acknowledged = 1 THEN 1 ELSE 0 END) AS review_acknowledged,
            SUM(CASE WHEN followup_completed = 1 THEN 1 ELSE 0 END) AS followup_completed
        FROM problems
        GROUP BY difficulty
        ORDER BY
            CASE lower(difficulty)
                WHEN 'easy' THEN 0
                WHEN 'medium' THEN 1
                WHEN 'hard' THEN 2
                ELSE 99
            END,
            lower(difficulty)
        """
    ).fetchall()
    return [
        {
            "difficulty": str(row["difficulty"]),
            "total": int(row["total"] or 0),
            "attempted": int(row["attempted"] or 0),
            "solved": int(row["solved"] or 0),
            "review_generated": int(row["review_generated"] or 0),
            "review_acknowledged": int(row["review_acknowledged"] or 0),
            "followup_completed": int(row["followup_completed"] or 0),
        }
        for row in rows
    ]


def fetch_weakest_tags_by_bundled_pass_rate(
    conn: sqlite3.Connection,
    *,
    limit: int = 5,
) -> list[dict[str, int | float | str]]:
    if limit <= 0:
        return []

    rows = conn.execute(
        """
        SELECT problems.tags, attempts.bundled_passed, attempts.bundled_total
        FROM attempts
        JOIN problems ON problems.slug = attempts.slug
        WHERE attempts.bundled_total > 0
        ORDER BY attempts.id ASC
        """
    ).fetchall()

    tag_stats: dict[str, dict[str, int | float | str]] = {}
    for row in rows:
        passed = int(row["bundled_passed"] or 0)
        total = int(row["bundled_total"] or 0)
        for tag in _problem_tags(row):
            entry = tag_stats.setdefault(
                tag,
                {
                    "tag": tag,
                    "attempts": 0,
                    "bundled_passed": 0,
                    "bundled_total": 0,
                    "pass_rate": 0.0,
                },
            )
            entry["attempts"] = int(entry["attempts"]) + 1
            entry["bundled_passed"] = int(entry["bundled_passed"]) + passed
            entry["bundled_total"] = int(entry["bundled_total"]) + total

    for entry in tag_stats.values():
        bundled_total = int(entry["bundled_total"])
        entry["pass_rate"] = (int(entry["bundled_passed"]) / bundled_total) if bundled_total else 0.0

    ordered = sorted(
        tag_stats.values(),
        key=lambda item: (
            float(item["pass_rate"]),
            -int(item["attempts"]),
            str(item["tag"]),
        ),
    )
    return ordered[:limit]


def fetch_problem_history_view(conn: sqlite3.Connection, slug: str) -> dict[str, Any] | None:
    problem = fetch_problem_record(conn, slug)
    if problem is None:
        return None

    attempts = conn.execute(
        """
        SELECT *
        FROM attempts
        WHERE slug = ?
        ORDER BY id DESC
        """,
        (slug,),
    ).fetchall()
    attempt_rows = [dict(row) for row in attempts]

    milestones = {
        "auto_solved": bool(problem.get("auto_solved", 0)),
        "auto_solved_at": problem.get("auto_solved_at"),
        "manual_solved": bool(problem.get("manual_solved", 0)),
        "manual_solved_at": problem.get("manual_solved_at"),
        "review_generated": bool(problem.get("review_generated", 0)),
        "review_generated_at": problem.get("review_generated_at"),
        "review_acknowledged": bool(problem.get("review_acknowledged", 0)),
        "review_acknowledged_at": problem.get("review_acknowledged_at"),
        "followup_completed": bool(problem.get("followup_completed", 0)),
        "followup_completed_at": problem.get("followup_completed_at"),
    }

    if not attempt_rows:
        return {
            "problem": problem,
            "milestones": milestones,
            "attempts": [],
        }

    attempt_ids = [int(row["id"]) for row in attempt_rows]
    placeholders = ", ".join(["?"] * len(attempt_ids))
    review_rows = conn.execute(
        f"""
        SELECT *
        FROM reviews
        WHERE attempt_id IN ({placeholders})
        """,
        attempt_ids,
    ).fetchall()
    reviews_by_attempt_id = {int(row["attempt_id"]): dict(row) for row in review_rows}
    review_ids = [int(row["id"]) for row in review_rows]
    extensions_by_review_id: dict[int, list[dict[str, Any]]] = {}
    if review_ids:
        review_placeholders = ", ".join(["?"] * len(review_ids))
        extension_rows = conn.execute(
            f"""
            SELECT *
            FROM extension_results
            WHERE review_id IN ({review_placeholders})
            ORDER BY id ASC
            """,
            review_ids,
        ).fetchall()
        for row in extension_rows:
            extensions_by_review_id.setdefault(int(row["review_id"]), []).append(dict(row))

    history_items: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempt_rows, start=1):
        review = reviews_by_attempt_id.get(int(attempt["id"]))
        review_id = int(review["id"]) if review is not None else None
        history_items.append(
            {
                "attempt_number": index,
                "attempt": attempt,
                "review": review,
                "extensions": extensions_by_review_id.get(review_id, []) if review_id is not None else [],
            }
        )

    return {
        "problem": problem,
        "milestones": milestones,
        "attempts": history_items,
    }


def select_random_unsolved_problem(
    conn: sqlite3.Connection,
    *,
    difficulty: str | None = None,
    tag: str | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any] | None:
    difficulty_filter = difficulty.strip().lower() if difficulty else None
    tag_filter = tag.strip().lower() if tag else None

    candidates: list[dict[str, Any]] = []
    for record in list_problem_records(conn):
        if int(record.get("auto_solved", 0) or 0) or int(record.get("manual_solved", 0) or 0):
            continue
        if difficulty_filter is not None and str(record.get("difficulty", "")).strip().lower() != difficulty_filter:
            continue
        if tag_filter is not None and tag_filter not in _problem_tags(record):
            continue
        candidates.append(record)

    if not candidates:
        logger.info("No unsolved problem matched difficulty=%s tag=%s", difficulty_filter, tag_filter)
        return None

    chooser = rng.choice if rng is not None else random.choice
    selected = chooser(candidates)
    logger.info(
        "Selected random unsolved problem %s difficulty=%s tag=%s",
        selected.get("slug"),
        difficulty_filter,
        tag_filter,
    )
    return selected
