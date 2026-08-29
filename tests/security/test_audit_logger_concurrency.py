"""
Tests for core.security.engine.audit_logger.AuditLogger's thread
safety (Build Phase 21: core.kernel.concurrent_kernel is the first
caller that can have more than one concurrently-running agent's own
tool calls writing security-decision events into the SAME
`audit_log_path` at once -- see _WRITE_LOCK's own docstring in
audit_logger.py for why this is a module-level lock, not a
per-instance one, mirroring core/memory/memory_store.py's own
_APPEND_LOCK).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.security.engine.audit_logger import AuditLogger


def test_concurrent_records_from_separate_logger_instances_never_corrupt_the_file():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        log_path = str(tmp_dir / "audit.jsonl")
        thread_count = 16
        records_per_thread = 10

        def _record_many(thread_index: int) -> None:
            # A fresh AuditLogger per thread, all pointed at the exact
            # same file -- exactly the shape this project's own
            # build_research/build_writer/build_reviewer factories
            # produce (each builds its own SecurityDecisionPoint/
            # AuditLogger from the same shared `audit_log_path`) once
            # more than one of them can run at once.
            logger = AuditLogger(log_path)
            for i in range(records_per_thread):
                logger.record(
                    {
                        "subject": f"thread-{thread_index}",
                        "event": "test_event",
                        "sequence": i,
                    }
                )

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            list(executor.map(_record_many, range(thread_count)))

        raw_lines = Path(log_path).read_text(encoding="utf-8").splitlines()

        # Every single record must have landed as its own valid,
        # parseable JSON line -- a lock failure would show up either
        # as fewer lines than expected (one writer's bytes silently
        # clobbered by another's) or a line that fails to parse (two
        # writers' bytes interleaved into one corrupt line).
        assert len(raw_lines) == thread_count * records_per_thread

        parsed = [json.loads(line) for line in raw_lines]
        assert len(parsed) == thread_count * records_per_thread
        assert all("timestamp" in entry for entry in parsed)

        sequences_seen = {
            (entry["subject"], entry["sequence"]) for entry in parsed
        }
        expected = {
            (f"thread-{t}", i)
            for t in range(thread_count)
            for i in range(records_per_thread)
        }
        assert sequences_seen == expected
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
