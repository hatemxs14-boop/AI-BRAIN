"""
Tests for Kernel(model_tier_router=...) -- Build Phase 27's Kernel-level
per-task model-tier routing wiring (core/llm/model_tier.py,
core/kernel/kernel.py).

Uses the same isolated, tempfile-based permissions.json fixture style
tests/kernel/test_kernel_budget.py/test_kernel_guardrails.py already
established, with a real, LOW-risk, auto-allowed "web_search" tool.
Unlike those two files, these tests need a REAL LLMDecisionEngine (not
a scripted AgentDecisionEngine test double) wired to a recording fake
LLMClient, since model-tier routing is applied specifically to
LLMDecisionEngine instances (see core/llm/model_tier.py's own module
docstring for why) -- a scripted decision engine has no `.model`
attribute at all and is simply never touched.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.checkpoint import FileCheckpointStore, TaskCheckpoint
from core.agents.deterministic_decision_engine import (
    DeterministicDecisionEngine,
)
from core.agents.llm_decision_engine import LLMDecisionEngine
from core.agents.tool_interface import AgentToolInterface

from core.kernel.kernel import AgentRegistration, Kernel

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.model_tier import ModelTierRouter

from core.orchestration.orchestration_engine import (
    SequentialOrchestrationEngine,
)

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _write_low_risk_search_policy(tmp_dir: Path, subject: str) -> Path:
    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": subject,
                "resource": "web_search",
                "action": "search",
                "scope": "public_web",
                "risk_level": "LOW",
                "approval": "none",
            }
        ],
        "defaults": {
            "unknown_risk": "DENY",
            "unknown_permission": "DENY",
            "unknown_scope": "DENY",
            "authorization_failure": "DENY",
        },
    }
    policy_path = tmp_dir / "permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return policy_path


def _build_low_risk_tool_agent(tmp_dir: Path, subject: str) -> AgentCore:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="web_search",
            name="Web Search",
            purpose="Search the public web.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "string"},
            permissions=(f"{subject}:web_search:search:public_web",),
            resource="web_search",
            action="search",
            scope="public_web",
            risk_level="LOW",
            error_handling={
                "retryable": True,
                "max_retries": 2,
                "on_failure": "Surface the search error to the agent.",
            },
        )
    )

    policy_path = _write_low_risk_search_policy(tmp_dir, subject=subject)

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(tmp_dir / "audit.jsonl"),
    )

    gateway = ToolGateway(security=security, registry=registry)
    gateway.register_executor(
        tool_id="web_search",
        executor=lambda query: f"RESULT: {query}",
    )

    runtime = ToolRuntime(registry=registry, gateway=gateway)
    interface = AgentToolInterface(runtime=runtime)

    identity = AgentIdentity(
        subject=subject,
        name="Test Agent",
        purpose="A minimal agent used only to exercise the Kernel's model tier router.",
    )

    return AgentCore(identity=identity, tools=interface)


class _RecordingLLMClient(LLMClient):
    """
    A minimal, real LLMClient that records the last LLMRequest it was
    asked to `generate()` -- specifically its own `.model` field -- so
    a test can confirm which model tier the Kernel's own
    `model_tier_router` actually routed a given task to. Always
    returns a clean COMPLETE action; mirrors tests/agents/
    test_llm_decision_engine.py's own MockLLMClient.
    """

    def __init__(self) -> None:
        self.last_request: LLMRequest | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(
            content=(
                '{"action_type":"COMPLETE","tool_id":null,'
                '"inputs":null,"reason":"Done."}'
            ),
            model=request.model or "unspecified",
            finish_reason="stop",
            usage=None,
        )


@pytest.fixture()
def tmp_dir():
    directory = Path(tempfile.mkdtemp())
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _kernel_with_llm_backed_agent(
    tmp_dir: Path,
    client: LLMClient,
    *,
    model_tier_router: ModelTierRouter | None = None,
) -> Kernel:
    kernel = Kernel(
        orchestration_engine=SequentialOrchestrationEngine(),
        model_tier_router=model_tier_router,
    )

    kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="A single agent backed by a real LLMDecisionEngine.",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(tmp_dir, "test_agent"),
            build_decision_engine=lambda: LLMDecisionEngine(
                client, model="default-model"
            ),
        )
    )

    return kernel


# ---------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------


def test_kernel_rejects_non_model_tier_router():
    with pytest.raises(TypeError, match="model_tier_router"):
        Kernel(model_tier_router="not-a-model-tier-router")


def test_kernel_without_model_tier_router_leaves_the_model_unchanged(tmp_dir):
    client = _RecordingLLMClient()
    kernel = _kernel_with_llm_backed_agent(tmp_dir, client)

    result = kernel.run("Search AI agents.")

    assert result.status == "COMPLETED"
    assert client.last_request is not None
    assert client.last_request.model == "default-model"


# ---------------------------------------------------------------------
# Enforcement -- run()
# ---------------------------------------------------------------------


def test_kernel_routes_a_simple_task_to_the_simple_model(tmp_dir):
    client = _RecordingLLMClient()
    kernel = _kernel_with_llm_backed_agent(
        tmp_dir,
        client,
        model_tier_router=ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
        ),
    )

    result = kernel.run("Search AI agents.")

    assert result.status == "COMPLETED"
    assert client.last_request is not None
    assert client.last_request.model == "cheap-model"


def test_kernel_routes_a_task_with_a_complexity_keyword_to_the_complex_model(
    tmp_dir,
):
    client = _RecordingLLMClient()
    kernel = _kernel_with_llm_backed_agent(
        tmp_dir,
        client,
        model_tier_router=ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
        ),
    )

    result = kernel.run("Please provide a comprehensive analysis of AI agents.")

    assert result.status == "COMPLETED"
    assert client.last_request is not None
    assert client.last_request.model == "expensive-model"


def test_kernel_routes_a_long_task_to_the_complex_model_by_word_count(tmp_dir):
    client = _RecordingLLMClient()
    kernel = _kernel_with_llm_backed_agent(
        tmp_dir,
        client,
        model_tier_router=ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
            simple_max_words=5,
        ),
    )

    result = kernel.run("This task has exactly seven plain words here.")

    assert result.status == "COMPLETED"
    assert client.last_request is not None
    assert client.last_request.model == "expensive-model"


def test_kernel_model_tier_router_does_not_affect_a_non_llm_decision_engine(
    tmp_dir,
):
    # DeterministicDecisionEngine has no `.model` attribute at all --
    # a configured model_tier_router must be a pure no-op for it, never
    # an error (the same tolerant shape guardrail_engine/token_budget
    # already established for a decision engine that doesn't expose
    # their own duck-typed attributes either).
    kernel = Kernel(
        orchestration_engine=SequentialOrchestrationEngine(),
        model_tier_router=ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
        ),
    )

    kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="A deterministic (non-LLM) agent.",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(tmp_dir, "test_agent"),
            build_decision_engine=DeterministicDecisionEngine,
        )
    )

    result = kernel.run("Search AI agents.")

    assert result.status == "COMPLETED"


# ---------------------------------------------------------------------
# Enforcement -- resume()
# ---------------------------------------------------------------------


def test_resume_also_applies_the_kernels_model_tier_routing(tmp_dir):
    client = _RecordingLLMClient()

    store = FileCheckpointStore(tmp_dir / "checkpoints")
    store.save(
        TaskCheckpoint(
            checkpoint_id="task-tier",
            subject="test_agent",
            task="Please provide a comprehensive analysis of AI agents.",
            step_count=1,
            tool_results=(
                {"status": "SUCCESS", "summary": "ok", "artifacts": []},
            ),
            last_tool_id="web_search",
        )
    )

    kernel = _kernel_with_llm_backed_agent(
        tmp_dir,
        client,
        model_tier_router=ModelTierRouter(
            simple_model="cheap-model",
            complex_model="expensive-model",
        ),
    )

    result = kernel.resume("task-tier", checkpoint_store=store)

    assert result.status == "COMPLETED"
    assert client.last_request is not None
    assert client.last_request.model == "expensive-model"
