from __future__ import annotations

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.token_usage import TokenUsage


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

    # Shared with OpenAIProvider so LLMRequest(max_tokens=None) behaves
    # identically regardless of which provider is configured, instead
    # of silently falling back to whatever each vendor SDK happens to
    # default to on its own (which can vary by model and change
    # between SDK versions).
    _DEFAULT_MAX_TOKENS = 1024

    # Anthropic's `stop_reason` values normalized into the small,
    # provider-independent vocabulary LLMResponse.finish_reason uses
    # (see OpenAIProvider._FINISH_REASON_MAP for the OpenAI side).
    # Callers that need the exact vendor value can still read it off
    # `LLMResponse.raw`.
    _FINISH_REASON_MAP = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_use",
    }

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
                else self._DEFAULT_MAX_TOKENS
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

        if not text_parts:
            # No text content at all (e.g. the response contains only
            # a tool_use block, or Anthropic returned an unexpectedly
            # empty content list). Previously this silently produced
            # `content=""`, which an empty response and a genuine
            # "the model said nothing" case both looked like -- the
            # caller only found out something was wrong indirectly,
            # several layers later, from a confusing JSON-parse
            # failure. OpenAIProvider already raises immediately for
            # its equivalent "no usable content" case; this keeps both
            # providers consistent instead of one failing loudly and
            # the other failing silently.
            raise ValueError(
                "Claude response contains no text content "
                f"(stop_reason={getattr(response, 'stop_reason', None)!r})."
            )

        content = "".join(text_parts)

        return LLMResponse(
            content=content,
            model=getattr(
                response,
                "model",
                request.model,
            ),
            finish_reason=self._normalize_finish_reason(
                getattr(response, "stop_reason", None)
            ),
            raw=response,
            usage=self._extract_usage(
                getattr(response, "usage", None)
            ),
        )

    @classmethod
    def _normalize_finish_reason(
        cls,
        raw_stop_reason: str | None,
    ) -> str | None:
        if raw_stop_reason is None:
            return None

        return cls._FINISH_REASON_MAP.get(raw_stop_reason, "other")

    @staticmethod
    def _extract_usage(raw_usage) -> TokenUsage | None:
        """
        Build a TokenUsage from Anthropic's own `response.usage`
        (an `input_tokens`/`output_tokens` object -- no `total_tokens`
        of its own, unlike OpenAI's, so it is computed here).

        `None` whenever `raw_usage` is missing either field or either
        field isn't a real int (a mock/test client with no `.usage`
        attribute at all, or an older/different SDK shape) -- this
        never fabricates a partial or zero usage.
        """

        input_tokens = getattr(raw_usage, "input_tokens", None)
        output_tokens = getattr(raw_usage, "output_tokens", None)

        if (
            isinstance(input_tokens, bool)
            or isinstance(output_tokens, bool)
            or not isinstance(input_tokens, int)
            or not isinstance(output_tokens, int)
        ):
            return None

        return TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
