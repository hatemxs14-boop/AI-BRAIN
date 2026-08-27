from __future__ import annotations

import pytest

from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)

from core.agents.agent_context import (
    AgentContext,
)

from core.agents.llm_decision_engine import (
    LLMDecisionEngine,
)

from core.llm.llm_client import (
    LLMClient,
)

from core.llm.llm_request import (
    LLMRequest,
)

from core.llm.llm_response import (
    LLMResponse,
)


class MockLLMClient(LLMClient):

    def __init__(
        self,
        response: str,
    ):
        self.response = response
        self.last_request: LLMRequest | None = None

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        self.last_request = request

        return LLMResponse(
            content=self.response,
            model="mock-model",
            finish_reason="stop",
        )


def test_llm_decision_engine_returns_tool_action():

    client = MockLLMClient(
        response=(
            '{"action_type":"INVOKE_TOOL",'
            '"tool_id":"web_search",'
            '"inputs":{"query":"AI agents"},'
            '"reason":"Research is required."}'
        )
    )

    engine = LLMDecisionEngine(
        client,
        model="mock-model",
        temperature=0.2,
        max_tokens=500,
    )

    context = AgentContext(
        task="AI agents"
    )

    action = engine.decide(context)

    assert isinstance(
        action,
        AgentAction,
    )

    assert (
        action.action_type
        == AgentActionType.INVOKE_TOOL
    )

    assert action.tool_id == "web_search"

    assert action.inputs == {
        "query": "AI agents"
    }

    assert action.reason == "Research is required."


def test_llm_decision_engine_builds_request_from_context():

    client = MockLLMClient(
        response=(
            '{"action_type":"COMPLETE",'
            '"tool_id":null,'
            '"inputs":null,'
            '"reason":"Task complete."}'
        )
    )

    engine = LLMDecisionEngine(
        client
    )

    context = AgentContext(
        task="Test task"
    )

    engine.decide(context)

    request = client.last_request

    assert request is not None

    assert len(
        request.messages
    ) == 2

    assert (
        request.messages[0].role
        == "system"
    )

    assert (
        request.messages[1].role
        == "user"
    )

    assert (
        "Test task"
        in request.messages[1].content
    )


def test_llm_decision_engine_returns_complete_action():

    client = MockLLMClient(
        response=(
            '{"action_type":"COMPLETE",'
            '"tool_id":null,'
            '"inputs":null,'
            '"reason":"Task completed successfully."}'
        )
    )

    engine = LLMDecisionEngine(
        client
    )

    context = AgentContext(
        task="Complete this task"
    )

    action = engine.decide(
        context
    )

    assert (
        action.action_type
        == AgentActionType.COMPLETE
    )

    assert action.tool_id is None

    assert action.inputs is None

    assert (
        action.reason
        == "Task completed successfully."
    )


def test_llm_decision_engine_returns_fail_action():

    client = MockLLMClient(
        response=(
            '{"action_type":"FAIL",'
            '"tool_id":null,'
            '"inputs":null,'
            '"reason":"Unable to continue."}'
        )
    )

    engine = LLMDecisionEngine(
        client
    )

    context = AgentContext(
        task="Impossible task"
    )

    action = engine.decide(
        context
    )

    assert (
        action.action_type
        == AgentActionType.FAIL
    )

    assert action.tool_id is None

    assert action.inputs is None

    assert (
        action.reason
        == "Unable to continue."
    )


def test_llm_decision_engine_rejects_invalid_json():

    client = MockLLMClient(
        response="THIS IS NOT JSON"
    )

    engine = LLMDecisionEngine(
        client
    )

    context = AgentContext(
        task="Test task"
    )

    with pytest.raises(
        ValueError,
        match="not valid JSON",
    ):
        engine.decide(
            context
        )


def test_llm_decision_engine_rejects_non_object_json():

    client = MockLLMClient(
        response='["INVALID"]'
    )

    engine = LLMDecisionEngine(
        client
    )

    context = AgentContext(
        task="Test task"
    )

    with pytest.raises(
        ValueError,
        match="must be a JSON object",
    ):
        engine.decide(
            context
        )


def test_llm_decision_engine_rejects_invalid_action_type():

    client = MockLLMClient(
        response=(
            '{"action_type":"HACK_SYSTEM",'
            '"tool_id":null,'
            '"inputs":null,'
            '"reason":"Malicious action."}'
        )
    )

    engine = LLMDecisionEngine(
        client
    )

    context = AgentContext(
        task="Test task"
    )

    with pytest.raises(
        ValueError,
        match="invalid action_type",
    ):
        engine.decide(
            context
        )


def test_llm_decision_engine_rejects_invalid_context():

    client = MockLLMClient(
        response=(
            '{"action_type":"COMPLETE",'
            '"tool_id":null,'
            '"inputs":null,'
            '"reason":"Done."}'
        )
    )

    engine = LLMDecisionEngine(
        client
    )

    with pytest.raises(
        TypeError,
        match="context must be an AgentContext",
    ):
        engine.decide(
            "INVALID_CONTEXT"
        )


def test_llm_decision_engine_rejects_invalid_client():

    with pytest.raises(
        TypeError,
        match="client must implement LLMClient",
    ):
        LLMDecisionEngine(
            "INVALID_CLIENT"
        )


def test_llm_decision_engine_passes_configuration():

    client = MockLLMClient(
        response=(
            '{"action_type":"COMPLETE",'
            '"tool_id":null,'
            '"inputs":null,'
            '"reason":"Done."}'
        )
    )

    engine = LLMDecisionEngine(
        client,
        model="test-model",
        temperature=0.7,
        max_tokens=123,
    )

    context = AgentContext(
        task="Configuration test"
    )

    engine.decide(
        context
    )

    request = client.last_request

    assert request is not None

    assert request.model == "test-model"

    assert request.temperature == 0.7

    assert request.max_tokens == 123


def test_llm_decision_engine_includes_tool_results_in_context():

    client = MockLLMClient(
        response=(
            '{"action_type":"COMPLETE",'
            '"tool_id":null,'
            '"inputs":null,'
            '"reason":"Research result received."}'
        )
    )

    engine = LLMDecisionEngine(
        client
    )

    context = AgentContext(
        task="Research AI"
    )

    context.record_tool_result(
        "SEARCH RESULT"
    )

    engine.decide(
        context
    )

    request = client.last_request

    assert request is not None

    assert (
        "SEARCH RESULT"
        in request.messages[1].content
    )