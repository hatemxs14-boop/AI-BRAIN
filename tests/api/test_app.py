"""
Tests for api/app.py (Build Phase 33, real UI Part 1): the actual
FastAPI HTTP layer.

`fastapi` is not installed in this sandbox (no PyPI access here, the
same situation as voyageai/deepeval/langfuse before it), so this
WHOLE file is skip-guarded with `pytest.importorskip("fastapi")` at
the top -- mirroring exactly how tests/llm/test_model_config.py
already treats `anthropic`/`openai`, and how
tests/observability/test_langfuse_trace.py's own "installed" test is
`pytest.importorskip`-guarded. Expected to report SKIPPED (the whole
file) here; expected to run for real once `fastapi`/`uvicorn` are
installed (`pip install fastapi uvicorn`).
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from api.app import create_app  # noqa: E402
from core.kernel.kernel import Kernel, KernelResult  # noqa: E402
from core.llm.token_usage import TokenUsage  # noqa: E402


class _StubKernel(Kernel):
    """See tests/api/test_service.py's own _StubKernel for the same
    "subclass Kernel, never call its __init__, override run()" shape
    this project already established for testing a thin orchestration
    layer above a real Kernel."""

    def __init__(self, result: KernelResult):
        self._result = result
        self.received_task: str | None = None

    def run(self, task: str, **kwargs) -> KernelResult:  # type: ignore[override]
        self.received_task = task
        return self._result


def _make_client(result: KernelResult | None = None, **create_app_kwargs) -> TestClient:
    if result is None:
        result = KernelResult(
            status="COMPLETED",
            subject="research_agent",
            loop_result=None,
            verification=None,
            reason="Done.",
            recovery_attempts=0,
            token_usage=TokenUsage(
                prompt_tokens=1, completion_tokens=2, total_tokens=3
            ),
        )

    app = create_app(kernel_factory=lambda: _StubKernel(result), **create_app_kwargs)
    return TestClient(app)


# ---------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------


def test_health_endpoint_returns_ok():
    client = _make_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------
# GET /system/status
# ---------------------------------------------------------------------


def test_system_status_endpoint_shape():
    client = _make_client()
    response = client.get("/system/status")

    assert response.status_code == 200
    body = response.json()
    assert "components" in body
    assert "all_configured" in body
    assert isinstance(body["components"], list)
    assert len(body["components"]) == 6
    for component in body["components"]:
        assert set(component.keys()) == {"name", "configured", "detail"}


# ---------------------------------------------------------------------
# GET /agents
# ---------------------------------------------------------------------


def test_agents_endpoint_returns_three_agents():
    client = _make_client()
    response = client.get("/agents")

    assert response.status_code == 200
    body = response.json()
    assert [a["subject"] for a in body] == [
        "research_agent",
        "writer_agent",
        "reviewer_agent",
    ]


# ---------------------------------------------------------------------
# POST /kernel/run
# ---------------------------------------------------------------------


def test_kernel_run_endpoint_returns_summary_from_the_real_stub_kernel():
    client = _make_client()
    response = client.post("/kernel/run", json={"task": "Research AI agents"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "COMPLETED"
    assert body["subject"] == "research_agent"
    assert body["reason"] == "Done."
    assert body["prompt_tokens"] == 1
    assert body["completion_tokens"] == 2
    assert body["total_tokens"] == 3


def test_kernel_run_endpoint_rejects_empty_task_with_400():
    client = _make_client()
    response = client.post("/kernel/run", json={"task": "   "})

    assert response.status_code == 400
    assert "task_text" in response.json()["detail"]


def test_kernel_run_endpoint_rejects_missing_task_field_with_422():
    # pydantic's own request-body validation, not api/service.py's --
    # this confirms the KernelRunRequest schema itself is wired in.
    client = _make_client()
    response = client.post("/kernel/run", json={})

    assert response.status_code == 422


def test_kernel_run_endpoint_returns_503_when_kernel_failed_to_build():
    app = create_app(kernel_factory=None, audit_log_path="/tmp/does-not-matter.jsonl")
    # Force the "Kernel failed to build" path deterministically,
    # without depending on this sandbox's real environment variables
    # (which may or may not have a real SERPER_API_KEY/model config
    # set) -- directly matching what create_app() itself does when
    # build_default_kernel() raises.
    app.state.kernel = None
    app.state.kernel_build_error = "simulated: SERPER_API_KEY not set"

    client = TestClient(app)
    response = client.post("/kernel/run", json={"task": "Research AI agents"})

    assert response.status_code == 503
    assert "SERPER_API_KEY not set" in response.json()["detail"]


# ---------------------------------------------------------------------
# GET /audit-log/recent
# ---------------------------------------------------------------------


def test_audit_log_recent_endpoint_returns_empty_list_for_missing_file(tmp_path):
    client = _make_client(audit_log_path=str(tmp_path / "no-such-file.jsonl"))
    response = client.get("/audit-log/recent")

    assert response.status_code == 200
    assert response.json() == []


def test_audit_log_recent_endpoint_reads_real_events(tmp_path):
    import json

    audit_log_path = tmp_path / "audit.jsonl"
    with open(audit_log_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps({"event": "a", "seq": 1}) + "\n")
        handle.write(json.dumps({"event": "b", "seq": 2}) + "\n")

    client = _make_client(audit_log_path=str(audit_log_path))
    response = client.get("/audit-log/recent")

    assert response.status_code == 200
    body = response.json()
    assert [e["seq"] for e in body] == [2, 1]


def test_audit_log_recent_endpoint_rejects_non_positive_limit(tmp_path):
    client = _make_client(audit_log_path=str(tmp_path / "no-such-file.jsonl"))
    response = client.get("/audit-log/recent", params={"limit": 0})

    assert response.status_code == 400


# ---------------------------------------------------------------------
# create_app()'s own graceful-degradation contract
# ---------------------------------------------------------------------


def test_create_app_default_path_never_raises_even_without_env_configured(
    monkeypatch,
):
    """
    The real production path (`kernel_factory=None`) must never crash
    app construction just because SERPER_API_KEY/model config aren't
    set in this process's environment -- see create_app()'s own
    docstring for why. This is the actual "system must never become
    so strict it refuses to run at all" standing constraint, verified
    for real, not just documented.
    """
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    app = create_app(audit_log_path="/tmp/does-not-matter-either.jsonl")

    assert app.state.kernel is None
    assert app.state.kernel_build_error is not None

    client = TestClient(app)
    # /health must still work with no working Kernel at all.
    assert client.get("/health").status_code == 200
