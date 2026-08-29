from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Module-level, not per-instance (Build Phase 21, mirroring
# core/memory/memory_store.py's own _APPEND_LOCK and its docstring's
# reasoning exactly): every agent's own tool stack builds its own
# AuditLogger instance from the same shared `audit_log_path`
# (core/kernel/default_kernel.py's build_research/build_writer/
# build_reviewer), so a per-instance lock would not serialize two
# concurrently-running agents' writers targeting the same file. Before
# this project had any concurrent/parallel request handling
# (ConcurrentKernelRunner, core/kernel/concurrent_kernel.py), two
# AuditLogger.record() calls could never race here in practice; now
# that a Kernel can run more than one task at once, every tool
# execution's audit entry goes through this same lock so two agents'
# entries can never interleave into one corrupted JSON line.
_WRITE_LOCK = threading.Lock()


class AuditLogger:
    """
    Append-only security audit logger.

    Every security decision can be recorded as a structured JSON event.
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }

        with _WRITE_LOCK:
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")