from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------
# Build Phase 14: the real (v1) Memory Layer described in
# core/memory/MEMORY_SPEC.md.
#
# Before this phase, no memory layer existed anywhere in this project
# -- see core/kernel/kernel.py's own module docstring ("no memory
# layer exists yet") and core/policies/policy_engine.py's own
# "MEMORY CONSTRAINTS -- No memory layer exists in this project" note.
# This module is that real foundation: a durable, append-only,
# secret-rejecting, verified/unverified-tracking store, kept
# deliberately narrow (keyword search, not a fabricated "semantic"
# layer -- see MEMORY_SPEC.md's own v1 Scope section for exactly what
# is and isn't built yet).
#
# Shape mirrors core/security/engine/audit_logger.py's own
# AuditLogger deliberately (append-only JSON Lines, a timestamp
# stamped on write, `parent.mkdir` at construction time) -- this is
# the same append-only pattern applied one layer up, for the same
# reason: a permanent, inspectable trail is more valuable than an
# updatable record, and this project already has one proven,
# regression-tested way to build that. It is a separate, independent
# module rather than a reuse of AuditLogger itself: POLICY_SPEC.md's
# own text says "Policies must remain separate from agents, tools,
# memory, and orchestration", and by the same reasoning the Memory
# Layer must remain independently replaceable from the Security
# Layer's own audit trail (SYSTEM_CONSTITUTION.md's Modularity rule) --
# reusing AuditLogger directly would quietly wire the two together.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryEntry:
    """
    What a caller supplies to MemoryStore.write() -- everything about
    one memory record except the id/timestamp the store itself
    assigns.

    `subject` is provenance: which agent or component this record is
    attributed to (SECURITY_SPEC.md's Memory Security: "provenance
    tracking"). `kind` is a free-form category (e.g. "note",
    "lesson") -- deliberately not a closed enum, since this project
    does not yet know the full set of memory kinds it will ever need,
    and a closed enum would need a code change for every new one.
    `verified` defaults to False -- POLICY_SPEC.md's Memory
    Constraints: "recalled memory is untrusted context" until
    something explicitly verifies it (see MemoryStore.verify()).
    """

    subject: str
    kind: str
    content: str
    verified: bool = False
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:

        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("MemoryEntry.subject must be a non-empty string.")

        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("MemoryEntry.kind must be a non-empty string.")

        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("MemoryEntry.content must be a non-empty string.")

        if not isinstance(self.verified, bool):
            raise TypeError("MemoryEntry.verified must be a bool.")

        if not isinstance(self.tags, tuple) or not all(
            isinstance(tag, str) for tag in self.tags
        ):
            raise TypeError("MemoryEntry.tags must be a tuple of strings.")


@dataclass(frozen=True)
class MemoryRecord:
    """
    A MemoryEntry as actually persisted: adds the id and timestamp the
    store assigns, plus (for a verification record produced by
    MemoryStore.verify()) `supersedes`, the id of the record it
    verifies. `supersedes` is `None` for an ordinary written record.
    """

    id: str
    timestamp: str
    subject: str
    kind: str
    content: str
    verified: bool
    tags: tuple[str, ...]
    supersedes: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "subject": self.subject,
            "kind": self.kind,
            "content": self.content,
            "verified": self.verified,
            "tags": list(self.tags),
            "supersedes": self.supersedes,
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> "MemoryRecord":
        return MemoryRecord(
            id=data["id"],
            timestamp=data["timestamp"],
            subject=data["subject"],
            kind=data["kind"],
            content=data["content"],
            verified=bool(data["verified"]),
            tags=tuple(data.get("tags") or ()),
            supersedes=data.get("supersedes"),
        )


# Finite, hand-maintained set of common secret shapes -- see
# MEMORY_SPEC.md's own "Secrets" section for why this is a real,
# best-effort defense and not a guarantee. Mirrors the honesty of
# RiskEngine's own sensitive-resource vocabulary
# (core/security/engine/risk_engine.py), which is likewise finite and
# hand-maintained rather than claimed complete.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI-style secret key
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(
        r"(?:api[_-]?key|secret|token|password|passwd)\s*[:=]\s*"
        r"['\"]?[A-Za-z0-9_\-./+]{12,}['\"]?",
        re.IGNORECASE,
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _looks_like_a_secret(content: str) -> bool:
    return any(pattern.search(content) for pattern in _SECRET_PATTERNS)


_TOKEN_PATTERN = re.compile(r"\w+")
_MIN_TOKEN_LENGTH = 3


def _tokenize(text: str) -> frozenset[str]:
    """
    Split `text` into lowercase word tokens, dropping anything shorter
    than `_MIN_TOKEN_LENGTH` -- short tokens ("a", "is", "to") match
    almost everything and would make search() effectively unfiltered.
    """

    return frozenset(
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if len(token) >= _MIN_TOKEN_LENGTH
    )


class MemoryStore:
    """
    Real (v1) implementation of core/memory/MEMORY_SPEC.md.

    Local, append-only, JSON-Lines-backed. `store_path`'s parent
    directory is created at construction time (mirrors
    AuditLogger.__init__'s own precedent) so a fresh deployment does
    not need to pre-create the directory by hand.
    """

    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: MemoryEntry) -> MemoryRecord:
        """
        Persist `entry` as a new record. Raises ValueError, refusing
        to write anything, when `entry.content` matches one of this
        module's known secret shapes -- POLICY_SPEC.md's Memory
        Constraints: "secrets ... must never be stored in AI-BRAIN
        memory." This fails loudly rather than silently storing a
        redacted placeholder (see MEMORY_SPEC.md's own "Secrets"
        section for why that distinction matters): a caller that
        genuinely needs to persist non-secret context about a
        credential (e.g. "the API key was rotated") can still do so
        as long as the actual secret value itself is not the content
        being written.
        """

        if not isinstance(entry, MemoryEntry):
            raise TypeError("entry must be a MemoryEntry.")

        if _looks_like_a_secret(entry.content):
            raise ValueError(
                "Refusing to write this memory entry: its content "
                "matches a known secret pattern. Secrets, credentials, "
                "tokens, and private keys must never be stored in "
                "AI-BRAIN memory (POLICY_SPEC.md's Memory Constraints; "
                "SECURITY_SPEC.md's Memory Security)."
            )

        record = MemoryRecord(
            id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            subject=entry.subject,
            kind=entry.kind,
            content=entry.content,
            verified=entry.verified,
            tags=entry.tags,
            supersedes=None,
        )

        self._append(record)

        return record

    def verify(self, record_id: str, *, verified_by: str) -> MemoryRecord:
        """
        Promote an existing record to verified by appending a NEW
        record that supersedes it, rather than mutating history --
        see MEMORY_SPEC.md's Record Shape section for why. Raises
        ValueError if `record_id` does not match any record currently
        in the store (nothing to verify).

        `verified_by` is the subject making the verification decision
        -- becomes the new record's own `subject`, so the trail shows
        who verified what, distinctly from who originally wrote it.
        """

        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string.")

        if not isinstance(verified_by, str) or not verified_by.strip():
            raise ValueError("verified_by must be a non-empty string.")

        original = self._find_by_id(record_id)

        if original is None:
            raise ValueError(
                f"No memory record with id {record_id!r} exists; "
                "nothing to verify."
            )

        record = MemoryRecord(
            id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            subject=verified_by,
            kind=original.kind,
            content=original.content,
            verified=True,
            tags=original.tags,
            supersedes=original.id,
        )

        self._append(record)

        return record

    def search(
        self,
        query: str,
        *,
        subject: str | None = None,
        verified_only: bool = False,
        limit: int = 10,
    ) -> tuple[MemoryRecord, ...]:
        """
        Simple, real (not fabricated-semantic) keyword match of
        `query` against each record's `content` -- see MEMORY_SPEC.md's
        v1 Scope for why keyword matching, not vector/semantic search,
        is the entire retrieval mechanism for now.

        Matching is word-based, not literal whole-string containment:
        `query` is split into lowercase word tokens (dropping anything
        shorter than 3 characters, which would otherwise match almost
        everything), and a record matches when its content contains
        ANY of those tokens as a substring -- e.g. a query of "the
        quarterly report" matches a record whose content is "the
        report is finished" because they share the token "report",
        even though neither string contains the other verbatim. A
        query with no token reaching that length (e.g. "AI") falls
        back to plain case-insensitive substring containment of the
        whole query, so a short-but-meaningful query still works.

        Returns the most recently written matches first, most recent
        verification records included (a verification record's own
        content is a copy of the original's, so it matches the same
        queries).

        `subject`, when given, restricts to records written by that
        exact subject. `verified_only`, when True, restricts to
        `verified=True` records -- the caller's way of asking only for
        canonical knowledge (POLICY_SPEC.md's Memory Constraints:
        "important information must be verified before becoming
        canonical knowledge").

        An empty or whitespace-only `query` raises ValueError rather
        than silently returning everything -- a caller that genuinely
        wants everything should say so explicitly by lowering `limit`
        appropriately, not rely on an empty string meaning "no
        filter", which would be easy to trigger by accident (e.g. an
        agent passing through an empty task string).
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")

        needle_tokens = _tokenize(query)
        lowered_query = query.lower()

        def _content_matches(content: str) -> bool:
            content_lower = content.lower()
            if needle_tokens:
                return any(token in content_lower for token in needle_tokens)
            return lowered_query in content_lower

        matches = [
            record
            for record in reversed(self._read_all())
            if _content_matches(record.content)
            and (subject is None or record.subject == subject)
            and (not verified_only or record.verified)
        ]

        return tuple(matches[:limit])

    # -- internals -------------------------------------------------

    def _append(self, record: MemoryRecord) -> None:
        with self.store_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def _read_all(self) -> tuple[MemoryRecord, ...]:
        if not self.store_path.exists():
            return ()

        lines = self.store_path.read_text(encoding="utf-8").splitlines()

        return tuple(
            MemoryRecord.from_dict(json.loads(line))
            for line in lines
            if line.strip()
        )

    def _find_by_id(self, record_id: str) -> MemoryRecord | None:
        for record in self._read_all():
            if record.id == record_id:
                return record
        return None
