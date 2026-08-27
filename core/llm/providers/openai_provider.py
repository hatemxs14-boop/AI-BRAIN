from __future__ import annotations

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse


class OpenAIProvider(LLMClient):
    """
    OpenAI implementation of the provider-independent LLMClient.

    This class only translates between the normalized AI-BRAIN
    LLM interface and the OpenAI API.
    """

    def __init__(self, client) -> None:
        self.client = client

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        if not isinstance(
            request,
            LLMRequest,
        ):
            raise TypeError(
                "request must be an LLMRequest."
            )

        if not request.messages:
            raise ValueError(
                "LLMRequest must contain at least one message."
            )

        if not isinstance(
            request.model,
            str,
        ) or not request.model.strip():
            raise ValueError(
                "LLMRequest must specify a model."
            )

        messages = []

        for message in request.messages:
            messages.append(
                {
                    "role": message.role,
                    "content": message.content,
                }
            )

        kwargs = {
            "model": request.model,
            "messages": messages,
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        response = self.client.chat.completions.create(
            **kwargs
        )

        if not getattr(response, "choices", None):
            raise ValueError(
                "OpenAI response contains no choices."
            )

        choice = response.choices[0]

        message = getattr(
            choice,
            "message",
            None,
        )

        content = getattr(
            message,
            "content",
            None,
        )

        if not isinstance(content, str):
            content = ""

        return LLMResponse(
            content=content,
            model=getattr(
                response,
                "model",
                request.model,
            ),
            finish_reason=getattr(
                choice,
                "finish_reason",
                None,
            ),
            raw=response,
        )
