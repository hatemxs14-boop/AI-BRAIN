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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.llm.embeddings import EmbeddingClient
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


# ---------------------------------------------------------------------
# Thread safety (Build Phase 21: core.kernel.concurrent_kernel is the
# first caller that can have more than one thread writing into the
# SAME underlying store_path at once -- e.g. two concurrently-running
# research_agent instances built from the same `memory_store_path`, per
# core/kernel/default_kernel.py's own build_research(). See _APPEND_
# LOCK's own docstring in core/memory/memory_store.py for why this is
# a module-level lock, not a per-instance one.)
# ---------------------------------------------------------------------


def test_concurrent_writes_from_separate_store_instances_never_corrupt_the_file():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store_path = str(tmp_dir / "memory.jsonl")
        thread_count = 16
        writes_per_thread = 10

        def _write_many(thread_index: int) -> None:
            # A fresh MemoryStore per thread, all pointed at the exact
            # same file -- exactly the shape build_research()'s own
            # factory produces under ConcurrentKernelRunner.
            store = MemoryStore(store_path)
            for i in range(writes_per_thread):
                store.write(
                    MemoryEntry(
                        subject=f"thread-{thread_index}",
                        kind="note",
                        content=f"entry {thread_index}-{i} some real content",
                    )
                )

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            list(executor.map(_write_many, range(thread_count)))

        # Every single write must have landed as its own valid,
        # parseable JSON line -- a lock failure here would show up as
        # fewer than expected lines (one writer's bytes silently
        # overwritten/interleaved into another's) or a line that
        # fails to parse (two writers' bytes merged into one corrupt
        # line).
        raw_lines = Path(store_path).read_text(encoding="utf-8").splitlines()
        assert len(raw_lines) == thread_count * writes_per_thread

        reader = MemoryStore(store_path)
        all_records = reader._read_all()
        assert len(all_records) == thread_count * writes_per_thread
        assert len({record.id for record in all_records}) == (
            thread_count * writes_per_thread
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# search_semantic() (Build Phase 28)
#
# Uses a fake, in-process EmbeddingClient double keyed by exact text ->
# vector, never a real network call -- exactly the same convention
# tests/llm/test_embeddings.py's own _FakeVoyageVendorClient
# establishes one layer down. Vectors are deliberately simple 2D unit-
# ish vectors chosen so cosine similarity works out to an exact,
# easy-to-assert value (1.0 for identical direction, 0.0 for
# orthogonal, -1.0 for opposite).
# ---------------------------------------------------------------------


class _FakeEmbeddingClient(EmbeddingClient):
    def __init__(self, vectors: dict) -> None:
        self._vectors = vectors
        self.calls: list = []

    def embed(self, texts, *, input_type):
        texts = tuple(texts)
        self.calls.append({"texts": texts, "input_type": input_type})
        return tuple(self._vectors[text] for text in texts)


def test_search_semantic_rejects_non_embedding_client():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        with pytest.raises(TypeError, match="EmbeddingClient"):
            store.search_semantic("query", embedding_client="not-a-client")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_rejects_empty_query():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        client = _FakeEmbeddingClient({})
        with pytest.raises(ValueError, match="query"):
            store.search_semantic("   ", embedding_client=client)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_rejects_non_positive_limit():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        client = _FakeEmbeddingClient({})
        with pytest.raises(ValueError, match="limit"):
            store.search_semantic("query", embedding_client=client, limit=0)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_rejects_non_numeric_min_similarity():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        client = _FakeEmbeddingClient({})
        with pytest.raises(TypeError, match="min_similarity"):
            store.search_semantic(
                "query", embedding_client=client, min_similarity="high"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_rejects_bool_min_similarity():
    # bool is a subclass of int -- must be rejected explicitly, the
    # same convention this project applies to every numeric field.
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        client = _FakeEmbeddingClient({})
        with pytest.raises(TypeError, match="min_similarity"):
            store.search_semantic(
                "query", embedding_client=client, min_similarity=True
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_on_empty_store_never_calls_embed():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        client = _FakeEmbeddingClient({})

        results = store.search_semantic("query", embedding_client=client)

        assert results == ()
        assert client.calls == []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_never_calls_embed_when_subject_filter_empties_the_pool():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="apple")
        )
        client = _FakeEmbeddingClient({})

        results = store.search_semantic(
            "query", embedding_client=client, subject="writer_agent"
        )

        assert results == ()
        assert client.calls == []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_filters_by_subject_before_embedding():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="apple")
        )
        store.write(
            MemoryEntry(subject="writer_agent", kind="note", content="banana")
        )
        client = _FakeEmbeddingClient(
            {"query": (1.0, 0.0), "banana": (1.0, 0.0)}
        )

        results = store.search_semantic(
            "query", embedding_client=client, subject="writer_agent"
        )

        assert len(results) == 1
        assert results[0].content == "banana"
        # Only the surviving candidate's content was ever embedded.
        document_call = next(
            call for call in client.calls if call["input_type"] == "document"
        )
        assert document_call["texts"] == ("banana",)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_verified_only_excludes_unverified_records():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        record = store.write(
            MemoryEntry(subject="research_agent", kind="finding", content="apple")
        )
        client = _FakeEmbeddingClient({"query": (1.0, 0.0), "apple": (1.0, 0.0)})

        assert (
            store.search_semantic(
                "query", embedding_client=client, verified_only=True
            )
            == ()
        )
        assert client.calls == []

        store.verify(record.id, verified_by="reviewer_agent")

        verified_results = store.search_semantic(
            "query", embedding_client=client, verified_only=True
        )
        assert len(verified_results) == 1
        assert verified_results[0].verified is True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_ranks_by_descending_cosine_similarity():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="carrot")
        )
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="banana")
        )
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="apple")
        )

        client = _FakeEmbeddingClient(
            {
                "query": (1.0, 0.0),
                "apple": (1.0, 0.0),  # cosine similarity 1.0
                "banana": (0.0, 1.0),  # cosine similarity 0.0
                "carrot": (-1.0, 0.0),  # cosine similarity -1.0
            }
        )

        # min_similarity=-1.0 so carrot's -1.0 similarity (below the
        # method's own default 0.0 threshold) is still included --
        # this test is about ranking order, not the threshold itself
        # (see test_search_semantic_respects_min_similarity_threshold
        # for that).
        results = store.search_semantic(
            "query", embedding_client=client, min_similarity=-1.0
        )

        assert [record.content for record in results] == [
            "apple",
            "banana",
            "carrot",
        ]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_respects_limit():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="carrot")
        )
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="banana")
        )
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="apple")
        )

        client = _FakeEmbeddingClient(
            {
                "query": (1.0, 0.0),
                "apple": (1.0, 0.0),
                "banana": (0.0, 1.0),
                "carrot": (-1.0, 0.0),
            }
        )

        results = store.search_semantic("query", embedding_client=client, limit=1)

        assert [record.content for record in results] == ["apple"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_respects_min_similarity_threshold():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="banana")
        )
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="apple")
        )

        client = _FakeEmbeddingClient(
            {
                "query": (1.0, 0.0),
                "apple": (1.0, 0.0),  # similarity 1.0
                "banana": (0.0, 1.0),  # similarity 0.0
            }
        )

        results = store.search_semantic(
            "query", embedding_client=client, min_similarity=0.5
        )

        assert [record.content for record in results] == ["apple"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_embeds_query_and_documents_with_correct_input_types():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        store.write(
            MemoryEntry(subject="research_agent", kind="note", content="apple")
        )
        client = _FakeEmbeddingClient({"query": (1.0, 0.0), "apple": (1.0, 0.0)})

        store.search_semantic("query", embedding_client=client)

        input_types = [call["input_type"] for call in client.calls]
        assert input_types == ["query", "document"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_search_semantic_makes_exactly_two_embed_calls_regardless_of_pool_size():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = _store(tmp_dir)
        vectors = {"query": (1.0, 0.0)}
        for i in range(10):
            content = f"item-{i}"
            store.write(
                MemoryEntry(
                    subject="research_agent", kind="note", content=content
                )
            )
            vectors[content] = (1.0, 0.0)

        client = _FakeEmbeddingClient(vectors)

        store.search_semantic("query", embedding_client=client, limit=3)

        assert len(client.calls) == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
