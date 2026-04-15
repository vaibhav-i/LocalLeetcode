from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re
import sqlite3
from typing import Any, Mapping

from . import db

_DEFAULT_VALIDATOR = "exact_match"
_VALIDATOR_ALIASES = {
    "exact": "exact_match",
    "exact_match": "exact_match",
    "set": "set_equality",
    "set_equality": "set_equality",
    "float": "float_tolerance",
    "float_tolerance": "float_tolerance",
    "any": "any_valid",
    "any_valid": "any_valid",
}


class ProblemParseError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class ProblemParam:
    name: str
    type: str


@dataclass(slots=True, frozen=True)
class ProblemScalingInputs:
    sizes: tuple[int, ...] = ()
    generator: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProblemScalingInputs":
        sizes_value = value.get("sizes", ())
        if isinstance(sizes_value, (list, tuple)):
            sizes = tuple(int(item) for item in sizes_value)
        elif sizes_value in (None, ""):
            sizes = ()
        else:
            raise ProblemParseError("scaling_inputs.sizes must be a list of integers")

        generator = str(value.get("generator", "") or "")
        extra = {
            key: item
            for key, item in value.items()
            if key not in {"sizes", "generator"}
        }
        return cls(sizes=sizes, generator=generator, extra=extra)

    def to_mapping(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.sizes:
            data["sizes"] = list(self.sizes)
        if self.generator:
            data["generator"] = self.generator
        data.update(self.extra)
        return data


@dataclass(slots=True, frozen=True)
class ProblemMetadata:
    schema_version: int
    title: str
    slug: str
    difficulty: str
    tags: tuple[str, ...]
    category: str
    function_name: str
    params: tuple[ProblemParam, ...]
    return_type: str
    validator: str = _DEFAULT_VALIDATOR
    scaling_inputs: ProblemScalingInputs | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ProblemMetadata":
        required = ("title", "slug", "difficulty", "function_name")
        missing = [key for key in required if key not in mapping]
        if missing:
            raise ProblemParseError(f"Missing required frontmatter fields: {', '.join(missing)}")

        schema_version = int(mapping.get("schema_version", 1))
        title = str(mapping["title"])
        slug = str(mapping["slug"])
        difficulty = str(mapping["difficulty"]).strip().lower()
        category = str(mapping.get("category", "") or "")
        function_name = str(mapping["function_name"])
        return_type = str(mapping.get("return_type", "") or "")

        tags_value = mapping.get("tags", ())
        if tags_value in (None, ""):
            tags: tuple[str, ...] = ()
        elif isinstance(tags_value, (list, tuple)):
            tags = tuple(str(tag) for tag in tags_value)
        else:
            tags = (str(tags_value),)

        params_value = mapping.get("params", ())
        params: list[ProblemParam] = []
        if params_value in (None, ""):
            params = []
        elif isinstance(params_value, (list, tuple)):
            for item in params_value:
                if not isinstance(item, Mapping):
                    raise ProblemParseError("Each params entry must be a mapping with name and type")
                if "name" not in item or "type" not in item:
                    raise ProblemParseError("Each params entry must include name and type")
                params.append(ProblemParam(name=str(item["name"]), type=str(item["type"])))
        else:
            raise ProblemParseError("params must be a list of mappings")

        validator = normalize_validator_name(str(mapping.get("validator", _DEFAULT_VALIDATOR)))

        scaling_inputs_value = mapping.get("scaling_inputs")
        scaling_inputs: ProblemScalingInputs | None = None
        if isinstance(scaling_inputs_value, Mapping):
            scaling_inputs = ProblemScalingInputs.from_mapping(scaling_inputs_value)
        elif scaling_inputs_value not in (None, ""):
            raise ProblemParseError("scaling_inputs must be a mapping when present")

        extras = {
            key: value
            for key, value in mapping.items()
            if key
            not in {
                "schema_version",
                "title",
                "slug",
                "difficulty",
                "tags",
                "category",
                "function_name",
                "params",
                "return_type",
                "validator",
                "scaling_inputs",
            }
        }

        return cls(
            schema_version=schema_version,
            title=title,
            slug=slug,
            difficulty=difficulty,
            tags=tags,
            category=category,
            function_name=function_name,
            params=tuple(params),
            return_type=return_type,
            validator=validator,
            scaling_inputs=scaling_inputs,
            extras=extras,
        )

    def to_mapping(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "title": self.title,
            "slug": self.slug,
            "difficulty": self.difficulty,
            "tags": list(self.tags),
            "category": self.category,
            "function_name": self.function_name,
            "params": [{"name": param.name, "type": param.type} for param in self.params],
            "return_type": self.return_type,
            "validator": self.validator,
        }
        if self.scaling_inputs is not None:
            data["scaling_inputs"] = self.scaling_inputs.to_mapping()
        data.update(self.extras)
        return data


@dataclass(slots=True, frozen=True)
class ProblemDocument:
    metadata: ProblemMetadata
    body: str
    problem_dir: Path
    statement_path: Path
    tests_path: Path | None
    starter_path: Path | None
    validator_path: Path | None
    reference_solution_path: Path | None
    statement_hash: str
    statement_mtime: float
    tests_hash: str | None
    tests_mtime: float | None

    @property
    def slug(self) -> str:
        return self.metadata.slug

    def to_db_record(self) -> dict[str, Any]:
        return {
            "slug": self.metadata.slug,
            "title": self.metadata.title,
            "difficulty": self.metadata.difficulty,
            "tags": db.json_dumps(list(self.metadata.tags)),
            "category": self.metadata.category,
            "function_name": self.metadata.function_name,
            "params": db.json_dumps([{"name": param.name, "type": param.type} for param in self.metadata.params]),
            "return_type": self.metadata.return_type,
            "validator": self.metadata.validator,
            "schema_version": self.metadata.schema_version,
            "file_path": str(self.problem_dir),
            "statement_hash": self.statement_hash,
            "statement_mtime": self.statement_mtime,
            "tests_hash": self.tests_hash,
            "tests_mtime": self.tests_mtime,
            "scaling_inputs": (
                db.json_dumps(self.metadata.scaling_inputs.to_mapping())
                if self.metadata.scaling_inputs is not None
                else None
            ),
        }


@dataclass(slots=True, frozen=True)
class IndexReport:
    scanned: int
    inserted: int
    updated: int
    skipped: int
    problems: tuple[ProblemDocument, ...]


def normalize_validator_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    return _VALIDATOR_ALIASES.get(normalized, normalized or _DEFAULT_VALIDATOR)


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _split_inline_items(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    for char in text:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            current.append(char)
            continue
        if char in "[{(":
            depth += 1
            current.append(char)
            continue
        if char in "]})":
            depth = max(0, depth - 1)
            current.append(char)
            continue
        if char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(char)
    trailing = "".join(current).strip()
    if trailing:
        items.append(trailing)
    return items


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    text = _strip_quotes(text)
    lowered = text.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?\d+\.\d+", text):
        return float(text)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in _split_inline_items(inner)]
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        if not inner:
            return {}
        mapping: dict[str, Any] = {}
        for item in _split_inline_items(inner):
            if ":" not in item:
                raise ProblemParseError(f"Invalid inline mapping item: {item}")
            key, raw_value = item.split(":", 1)
            mapping[key.strip()] = _parse_scalar(raw_value.strip())
        return mapping
    return text


def _parse_block(lines: list[str], start_index: int, indent: int) -> tuple[Any, int]:
    index = start_index
    mapping: dict[str, Any] = {}
    sequence: list[Any] = []
    mode: str | None = None

    while index < len(lines):
        line = lines[index]
        if _is_comment_or_blank(line):
            index += 1
            continue

        current_indent = _leading_spaces(line)
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ProblemParseError(f"Unexpected indentation on line {index + 1}")

        stripped = line.strip()
        if stripped.startswith("- "):
            if mode == "mapping":
                raise ProblemParseError(f"Cannot mix mappings and sequences at line {index + 1}")
            mode = "sequence"
            item, index = _parse_sequence_item(lines, index, indent)
            sequence.append(item)
            continue

        if ":" not in stripped:
            raise ProblemParseError(f"Expected a key/value pair on line {index + 1}")

        if mode == "sequence":
            raise ProblemParseError(f"Cannot mix mappings and sequences at line {index + 1}")
        mode = "mapping"

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1

        if raw_value:
            mapping[key] = _parse_scalar(raw_value)
            continue

        next_index = _next_significant_line(lines, index)
        if next_index is None:
            mapping[key] = None
            continue

        next_indent = _leading_spaces(lines[next_index])
        if next_indent <= indent:
            mapping[key] = None
            continue

        value, index = _parse_block(lines, next_index, next_indent)
        mapping[key] = value

    if mode == "sequence":
        return sequence, index
    return mapping, index


def _parse_sequence_item(lines: list[str], start_index: int, indent: int) -> tuple[Any, int]:
    line = lines[start_index]
    content = line.strip()[2:].strip()
    index = start_index + 1

    if not content:
        next_index = _next_significant_line(lines, index)
        if next_index is None:
            return None, index
        next_indent = _leading_spaces(lines[next_index])
        if next_indent <= indent:
            return None, index
        value, index = _parse_block(lines, next_index, next_indent)
        return value, index

    if ":" in content and not content.startswith(("[", "{")):
        key, raw_value = content.split(":", 1)
        item: dict[str, Any] = {key.strip(): _parse_scalar(raw_value.strip()) if raw_value.strip() else None}
        next_index = _next_significant_line(lines, index)
        if next_index is not None:
            next_indent = _leading_spaces(lines[next_index])
            if next_indent > indent:
                nested, index = _parse_block(lines, next_index, next_indent)
                if isinstance(nested, Mapping):
                    item.update(nested)
                else:
                    raise ProblemParseError(
                        f"Expected mapping content for list item starting on line {start_index + 1}"
                    )
        return item, index

    return _parse_scalar(content), index


def _next_significant_line(lines: list[str], start_index: int) -> int | None:
    for index in range(start_index, len(lines)):
        if not _is_comment_or_blank(lines[index]):
            return index
    return None


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines:
        raise ProblemParseError("Statement file is empty")

    start_index = _next_significant_line(lines, 0)
    if start_index is None or lines[start_index].strip() != "---":
        raise ProblemParseError("Missing YAML frontmatter start delimiter ('---')")

    end_index = None
    for index in range(start_index + 1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index is None:
        raise ProblemParseError("Missing YAML frontmatter end delimiter ('---')")

    frontmatter_lines = lines[start_index + 1 : end_index]
    frontmatter, _ = _parse_block(frontmatter_lines, 0, 0)
    if not isinstance(frontmatter, Mapping):
        raise ProblemParseError("Frontmatter must parse to a mapping")

    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    return dict(frontmatter), body


def parse_statement(statement_path: str | Path) -> ProblemDocument:
    statement_path = Path(statement_path)
    text = statement_path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(text)
    metadata = ProblemMetadata.from_mapping(frontmatter)

    problem_dir = statement_path.parent
    tests_path = problem_dir / "tests.json"
    starter_path = problem_dir / "starter.py"
    validator_path = problem_dir / "validator.py"
    reference_solution_path = problem_dir / "solutions" / "reference.py"

    statement_bytes = text.encode("utf-8")
    tests_bytes: bytes | None = None
    if tests_path.exists():
        tests_bytes = tests_path.read_bytes()

    return ProblemDocument(
        metadata=metadata,
        body=body,
        problem_dir=problem_dir.resolve(),
        statement_path=statement_path.resolve(),
        tests_path=tests_path.resolve() if tests_path.exists() else None,
        starter_path=starter_path.resolve() if starter_path.exists() else None,
        validator_path=validator_path.resolve() if validator_path.exists() else None,
        reference_solution_path=reference_solution_path.resolve() if reference_solution_path.exists() else None,
        statement_hash=_sha256_bytes(statement_bytes),
        statement_mtime=statement_path.stat().st_mtime,
        tests_hash=_sha256_bytes(tests_bytes) if tests_bytes is not None else None,
        tests_mtime=tests_path.stat().st_mtime if tests_path.exists() else None,
    )


def parse_problem_directory(problem_dir: str | Path) -> ProblemDocument:
    problem_dir = Path(problem_dir)
    statement_path = problem_dir / "statement.md"
    if not statement_path.exists():
        raise FileNotFoundError(f"No statement.md found in {problem_dir}")
    document = parse_statement(statement_path)
    if document.slug != problem_dir.name:
        raise ProblemParseError(
            f"Problem slug '{document.slug}' does not match directory name '{problem_dir.name}'"
        )
    return document


def discover_problem_directories(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    if (root / "statement.md").exists():
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir())


def should_reindex(existing_record: Mapping[str, Any] | None, document: ProblemDocument) -> bool:
    if existing_record is None:
        return True
    return not (
        existing_record.get("statement_hash") == document.statement_hash
        and float(existing_record.get("statement_mtime") or 0) == document.statement_mtime
        and existing_record.get("tests_hash") == document.tests_hash
        and float(existing_record.get("tests_mtime") or 0) == (document.tests_mtime or 0)
    )


def index_problem_bank(
    conn: sqlite3.Connection,
    problems_root: str | Path,
    *,
    prune_missing: bool = False,
) -> IndexReport:
    problem_dirs = discover_problem_directories(problems_root)
    scanned = inserted = updated = skipped = 0
    documents: list[ProblemDocument] = []
    seen_slugs: set[str] = set()

    for problem_dir in problem_dirs:
        statement_path = problem_dir / "statement.md"
        if not statement_path.exists():
            continue
        document = parse_problem_directory(problem_dir)
        scanned += 1
        documents.append(document)
        seen_slugs.add(document.slug)

        existing = db.fetch_problem_record(conn, document.slug)
        if not should_reindex(existing, document):
            skipped += 1
            continue

        if existing is None:
            inserted += 1
        else:
            updated += 1

        db.upsert_problem_record(conn, document.to_db_record())

    if prune_missing:
        existing_slugs = {
            str(row["slug"])
            for row in conn.execute("SELECT slug FROM problems").fetchall()
        }
        stale_slugs = sorted(existing_slugs - seen_slugs)
        if stale_slugs:
            db.delete_problem_records(conn, stale_slugs)

    return IndexReport(
        scanned=scanned,
        inserted=inserted,
        updated=updated,
        skipped=skipped,
        problems=tuple(documents),
    )


def load_problem_by_slug(conn: sqlite3.Connection, slug: str) -> ProblemDocument | None:
    row = db.fetch_problem_record(conn, slug)
    if row is None:
        return None

    problem_dir = Path(row["file_path"])
    statement_path = problem_dir / "statement.md"
    if not statement_path.exists():
        return None

    document = parse_statement(statement_path)
    if document.slug != slug:
        raise ProblemParseError(f"Problem directory for slug '{slug}' is inconsistent with frontmatter")
    return document


def problem_documents_from_root(root: str | Path) -> list[ProblemDocument]:
    documents: list[ProblemDocument] = []
    for problem_dir in discover_problem_directories(root):
        statement_path = problem_dir / "statement.md"
        if statement_path.exists():
            documents.append(parse_problem_directory(problem_dir))
    return documents


def problem_record_to_metadata(record: Mapping[str, Any]) -> ProblemMetadata:
    params_raw = record.get("params", "[]")
    if isinstance(params_raw, str) and params_raw:
        params_data = json.loads(params_raw)
    elif isinstance(params_raw, list):
        params_data = params_raw
    else:
        params_data = []

    scaling_inputs_raw = record.get("scaling_inputs")
    scaling_inputs = None
    if isinstance(scaling_inputs_raw, str) and scaling_inputs_raw:
        scaling_inputs = ProblemScalingInputs.from_mapping(json.loads(scaling_inputs_raw))
    elif isinstance(scaling_inputs_raw, Mapping):
        scaling_inputs = ProblemScalingInputs.from_mapping(scaling_inputs_raw)

    tags_raw = record.get("tags", "[]")
    if isinstance(tags_raw, str) and tags_raw:
        tags_data = json.loads(tags_raw)
    elif isinstance(tags_raw, list):
        tags_data = tags_raw
    else:
        tags_data = []

    return ProblemMetadata(
        schema_version=int(record.get("schema_version", 1)),
        title=str(record.get("title", "")),
        slug=str(record.get("slug", "")),
        difficulty=str(record.get("difficulty", "")),
        tags=tuple(str(item) for item in tags_data),
        category=str(record.get("category", "")),
        function_name=str(record.get("function_name", "")),
        params=tuple(
            ProblemParam(name=str(item["name"]), type=str(item["type"]))
            for item in params_data
            if isinstance(item, Mapping) and "name" in item and "type" in item
        ),
        return_type=str(record.get("return_type", "")),
        validator=normalize_validator_name(str(record.get("validator", _DEFAULT_VALIDATOR))),
        scaling_inputs=scaling_inputs,
    )


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
