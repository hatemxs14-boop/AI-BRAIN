from __future__ import annotations

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse


class ClaudeProvider(LLMClient):
    """
    Anthropic Claude implementation of the provider-independent LLMClient.

    This class is responsible only for translating between the
    provider-independent AI-BRAIN LLM interface and Anthropic's API.

    It does not:

    - decide agent actions
    - execute tools
    - authorize operations
    - access the Security Layer
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

        system_parts = []
        messages = []

        for message in request.messages:

            if message.role == "system":
                system_parts.append(
                    message.content
                )
            else:
                messages.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )

        if not messages:
            raise ValueError(
                "LLMRequest must contain at least one "
                "non-system message."
            )

        kwargs = {
            "model": request.model,
            "max_tokens": (
                request.max_tokens
                if request.max_tokens is not None
                else 1024
            ),
            "messages": messages,
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if system_parts:
            kwargs["system"] = "\n\n".join(
                system_parts
            )

        response = self.client.messages.create(
            **kwargs
        )

        text_parts = []

        for block in getattr(
            response,
            "content",
            [],
        ):

            text = getattr(
                block,
                "text",
                None,
            )

            if isinstance(text, str):
                text_parts.append(text)

        content = "".join(text_parts)

        return LLMResponse(
            content=content,
            model=getattr(
                response,
                "model",
                request.model,
            ),
            finish_reason=getattr(
                response,
                "stop_reason",
                None,
            ),
            raw=response,
        )
