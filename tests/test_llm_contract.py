from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMMessage, LLMRequest
from core.llm.llm_response import LLMResponse


def test_llm_request():
    request = LLMRequest(
        messages=(
            LLMMessage(
                role="user",
                content="Hello",
            ),
        ),
        model="test-model",
        temperature=0.2,
        max_tokens=100,
    )

    assert request.messages[0].role == "user"
    assert request.messages[0].content == "Hello"
    assert request.model == "test-model"


def test_llm_response():
    response = LLMResponse(
        content="Hello back",
        model="test-model",
        finish_reason="stop",
    )

    assert response.content == "Hello back"
    assert response.model == "test-model"
    assert response.finish_reason == "stop"


def test_llm_client_is_abstract():
    try:
        LLMClient()
        assert False, "LLMClient must be abstract."
    except TypeError:
        pass
