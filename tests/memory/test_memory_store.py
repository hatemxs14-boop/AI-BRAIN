"""
Tests for core.memory.memory_store (Build Phase 14: the real Memory
Layer described in core/memory/MEMORY_SPEC.md).

Covers: append-only persistence and reopening the same file fresh,
the verified/unverified distinction and verify()'s supersede-not-
mutate behavior, write-time secret rejection, and search()'s
filtering/ordering/limit rules.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from core.memory.memory_store import MemoryEntry, MemoryRecord, MemoryStore


def _store(tmp_dir: Path) -> MemoryStore:
    return MemoryStore(str(tmp_dir / "memory.jsonl"))


# ---------------------------------------------------------------------
# MemoryEntry validation
# ---------------------------------------------------------------------

def test_memory_entry_rejects_empty_subject():
    with pytest.raises(ValueError, match="subject"):
        MemoryEntry(subject="", kind="note", content="hello")


def test_memory_entry_rejects_empty_kind():
    with pytest.raises(ValueError, match="kind"):
        MemoryEntry(subject="research_agent", kind="", content="hello")


def test_memory_entry_rejects_empty_content():
    with pytest.raises(ValueError, match="content"):
        MemoryEntry(subject="research_agent", kind="note", content="")


def test_memory_entry_defaults_to_unverified():
    entry = MemoryEntry(subject="research_agent", kind="note", content="hello")
    assert entry.verified is False
    assert entry.tags == ()


# ---------------------------------------------------------------------
# write() / persistence
# ---------------------------------------------------------------------

def test_write_then_search_round_trips_content():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)

        record = store.write(
            MemoryEntry(
                subject="research_agent",
                kind="note",
                content="The Eiffel Tower is in Paris.",
            )
        )

        assert isinstance(record, MemoryRecord)
        assert record.verified is False
        assert record.id
        assert record.timestamp

        (found,) = store.search("eiffel")
        assert found.id == record.id
        assert found.content == "The Eiffel Tower is in Paris."
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_store_creates_its_parent_directory():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        nested_path = tmp_dir / "a" / "b" / "memory.jsonl"
        store = MemoryStore(str(nested_path))

        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="hi")
        )

        assert nested_path.exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_a_fresh_store_instance_reads_records_written_by_a_prior_one():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = str(tmp_dir / "memory.jsonl")

        MemoryStore(path).write(
            MemoryEntry(
                subject="research_agent",
                kind="note",
                content="Persisted across instances.",
            )
        )

        reopened = MemoryStore(path)
        (found,) = reopened.search("persisted")
        assert found.content == "Persisted across instances."
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_on_an_empty_store_returns_nothing():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        assert store.search("anything") == ()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_write_rejects_non_memory_entry():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        with pytest.raises(TypeError, match="MemoryEntry"):
            store.write("not an entry")  # type: ignore[arg-type]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# Secret rejection
# ---------------------------------------------------------------------

_KNOWN_SECRET_SHAPES: tuple[str, ...] = (
    "Here is the key: sk-abcdefghijklmnopqrstuvwxyz123456",
    "AWS access key AKIAABCDEFGHIJKLMNOP for the prod account.",
    "api_key: 'sup3r-long-secret-value-123456'",
    "token=abcdef0123456789abcdef0123456789",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIB...",
)


def test_write_rejects_content_matching_a_known_secret_shape():
    # A plain loop over cases, not pytest.mark.parametrize -- this
    # project's own test harness (see the throwaway pytest shim used
    # to run this suite) does not expand parametrized cases, and no
    # other test in this project uses it either.
    for secret_content in _KNOWN_SECRET_SHAPES:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            store = _store(tmp_dir)
            with pytest.raises(ValueError, match="secret"):
                store.write(
                    MemoryEntry(
                        subject="research_agent",
                        kind="note",
                        content=secret_content,
                    )
                )
            # Nothing should have been written at all -- the store's
            # backing file must not even exist yet.
            assert not Path(store.store_path).exists(), (
                f"secret shape leaked through: {secret_content!r}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def test_write_allows_ordinary_content_mentioning_the_word_secret():
    """
    Non-regression: this project's standing "never so strict it can't
    execute" constraint applies here too -- a note that merely
    discusses secrets in the abstract, without containing an actual
    secret-shaped value, must not be blocked.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)

        record = store.write(
            MemoryEntry(
                subject="research_agent",
                kind="note",
                content=(
                    "The OpenAI API key was rotated today; the old one "
                    "is no longer valid."
                ),
            )
        )

        assert record.content.startswith("The OpenAI API key")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------

def test_verify_appends_a_new_record_rather_than_mutating_the_original():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)

        original = store.write(
            MemoryEntry(
                subject="research_agent",
                kind="finding",
                content="Confirmed by two independent sources.",
            )
        )

        verified = store.verify(original.id, verified_by="reviewer_agent")

        assert verified.id != original.id
        assert verified.verified is True
        assert verified.supersedes == original.id
        assert verified.subject == "reviewer_agent"
        assert verified.content == original.content

        # The original record itself is untouched.
        all_records = store._read_all()  # noqa: SLF001 -- internal, test-only
        original_after = next(r for r in all_records if r.id == original.id)
        assert original_after.verified is False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_verify_raises_for_an_unknown_record_id():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        with pytest.raises(ValueError, match="No memory record"):
            store.verify("does-not-exist", verified_by="reviewer_agent")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_verify_rejects_empty_verified_by():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        original = store.write(
            MemoryEntry(subject="research_agent", kind="note", content="x")
        )
        with pytest.raises(ValueError, match="verified_by"):
            store.verify(original.id, verified_by="")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# search() filtering, ordering, limit
# ---------------------------------------------------------------------

def test_search_is_case_insensitive_substring_match():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="Paris is lovely.")
        )
        assert len(store.search("PARIS")) == 1
        assert len(store.search("tokyo")) == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_rejects_empty_query():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        with pytest.raises(ValueError, match="query"):
            store.search("")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_rejects_non_positive_limit():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        with pytest.raises(ValueError, match="limit"):
            store.search("anything", limit=0)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_returns_most_recently_written_matches_first():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        first = store.write(
            MemoryEntry(subject="research_agent", kind="note", content="apple one")
        )
        second = store.write(
            MemoryEntry(subject="research_agent", kind="note", content="apple two")
        )

        results = store.search("apple")

        assert [r.id for r in results] == [second.id, first.id]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_respects_limit():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        for i in range(5):
            store.write(
                MemoryEntry(
                    subject="research_agent",
                    kind="note",
                    content=f"apple {i}",
                )
            )

        assert len(store.search("apple", limit=2)) == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_filters_by_subject():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="apple from research")
        )
        store.write(
            MemoryEntry(subject="writer_agent", kind="note", content="apple from writer")
        )

        results = store.search("apple", subject="writer_agent")

        assert len(results) == 1
        assert results[0].subject == "writer_agent"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_verified_only_excludes_unverified_records():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        record = store.write(
            MemoryEntry(subject="research_agent", kind="finding", content="apple finding")
        )

        assert store.search("apple", verified_only=True) == ()

        store.verify(record.id, verified_by="reviewer_agent")

        verified_results = store.search("apple", verified_only=True)
        assert len(verified_results) == 1
        assert verified_results[0].verified is True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_tags_round_trip():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        store.write(
            MemoryEntry(
                subject="research_agent",
                kind="note",
                content="apple with tags",
                tags=("fruit", "example"),
            )
        )

        (found,) = store.search("apple")
        assert found.tags == ("fruit", "example")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
