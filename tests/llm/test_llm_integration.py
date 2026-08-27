from __future__ import annotations

from core.agents.agent_action import AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.llm_decision_engine import LLMDecisionEngine

from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse

from core.llm.providers.claude_provider import ClaudeProvider
from core.llm.providers.openai_provider import OpenAIProvider


class MockClaudeMessages:

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_kwargs = None

    def create(self, **kwargs):

        self.last_kwargs = kwargs

        class Block:
            def __init__(self, text):
                self.text = text

        class Response:
            content = [Block(self.response_text)]
            model = "claude-integration-test"
            stop_reason = "end_turn"

        return Response()


class MockClaudeClient:

    def __init__(self, response_text: str):
        self.messages = MockClaudeMessages(
            response_text
        )


class MockOpenAICompletions:

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_kwargs = None

    def create(self, **kwargs):

        self.last_kwargs = kwargs

        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, content):
                self.message = Message(content)
                self.finish_reason = "stop"

        class Response:
            choices = [
                Choice(self.response_text)
            ]
            model = "openai-integration-test"

        return Response()


class MockOpenAIChat:

    def __init__(self, response_text: str):
        self.completions = MockOpenAICompletions(
            response_text
        )


class MockOpenAIClient:

    def __init__(self, response_text: str):
        self.chat = MockOpenAIChat(
            response_text
        )


def test_claude_full_llm_decision_path():

    llm_action = (
        '{"action_type":"INVOKE_TOOL",'
        '"tool_id":"web_search",'
        '"inputs":{"query":"AI agents"},'
        '"reason":"Research is required."}'
    )

    client = MockClaudeClient(
        response_text=llm_action
    )

    provider = ClaudeProvider(
        client=client
    )

    engine = LLMDecisionEngine(
        provider,
        model="claude-integration-test",
    )

    context = AgentContext(
        task="AI agents"
    )

    action = engine.decide(context)

    assert action.action_type == (
        AgentActionType.INVOKE_TOOL
    )

    assert action.tool_id == "web_search"

    assert action.inputs == {
        "query": "AI agents"
    }

    assert action.reason == (
        "Research is required."
    )


def test_openai_full_llm_decision_path():

    llm_action = (
        '{"action_type":"COMPLETE",'
        '"reason":"Task completed."}'
    )

    client = MockOpenAIClient(
        response_text=llm_action
    )

    provider = OpenAIProvider(
        client=client
    )

    engine = LLMDecisionEngine(
        provider,
        model="openai-integration-test",
    )

    context = AgentContext(
        task="Complete task"
    )

    action = engine.decide(context)

    assert action.action_type == (
        AgentActionType.COMPLETE
    )

    assert action.tool_id is None

    assert action.inputs is None

    assert action.reason == (
        "Task completed."
    )


def test_claude_provider_receives_decision_engine_request():

    llm_action = (
        '{"action_type":"COMPLETE",'
        '"reason":"Done."}'
    )

    client = MockClaudeClient(
        response_text=llm_action
    )

    provider = ClaudeProvider(
        client=client
    )

    engine = LLMDecisionEngine(
        provider,
        model="claude-integration-test",
        temperature=0.3,
        max_tokens=400,
    )

    context = AgentContext(
        task="Integration test"
    )

    engine.decide(context)

    kwargs = client.messages.last_kwargs

    assert kwargs is not None

    assert kwargs["model"] == (
        "claude-integration-test"
    )

    assert kwargs["temperature"] == 0.3

    assert kwargs["max_tokens"] == 400

    assert (
        kwargs["messages"][0]["role"]
        == "user"
    )

    assert (
        "Integration test"
        in kwargs["messages"][0]["content"]
    )


def test_openai_provider_receives_decision_engine_request():

    llm_action = (
        '{"action_type":"COMPLETE",'
        '"reason":"Done."}'
    )

    client = MockOpenAIClient(
        response_text=llm_action
    )

    provider = OpenAIProvider(
        client=client
    )

    engine = LLMDecisionEngine(
        provider,
        model="openai-integration-test",
        temperature=0.4,
        max_tokens=300,
    )

    context = AgentContext(
        task="OpenAI integration"
    )

    engine.decide(context)

    kwargs = (
        client.chat.completions.last_kwargs
    )

    assert kwargs is not None

    assert kwargs["model"] == (
        "openai-integration-test"
    )

    assert kwargs["temperature"] == 0.4

    assert kwargs["max_tokens"] == 300

    assert kwargs["messages"][0] == {
        "role": "system",
        "content": (
            "You are an AI-BRAIN agent decision engine. "
            "Return exactly one JSON object describing the "
            "next AgentAction. "
            "Do not execute tools yourself."
        ),
    }

    assert kwargs["messages"][1] == {
        "role": "user",
        "content": (
            "Current agent context:\n"
            '{"task": "OpenAI integration", '
            '"tool_results": []}\n\n'
            "Return one action as JSON."
        ),
    }
