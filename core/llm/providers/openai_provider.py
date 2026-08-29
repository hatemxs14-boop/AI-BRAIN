from __future__ import annotations

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.token_usage import TokenUsage


class OpenAIProvider(LLMClient):
    """
    OpenAI implementation of the provider-independent LLMClient.

    This class only translates between the normalized AI-BRAIN
    LLM interface and the OpenAI API.
    """

    # Shared in spirit with ClaudeProvider._DEFAULT_MAX_TOKENS: without
    # an explicit default here, LLMRequest(max_tokens=None) previously
    # omitted the parameter entirely and let each model's own OpenAI
    # default apply -- which is vendor- and model-specific and can
    # change between API/SDK versions, unlike Claude's provider, which
    # already applied an explicit default. Using the same value for
    # both providers means the *same* LLMRequest behaves identically
    # regardless of which provider is configured.
    _DEFAULT_MAX_TOKENS = 1024

    # OpenAI's `finish_reason` values normalized into the small,
    # provider-independent vocabulary LLMResponse.finish_reason uses
    # (see ClaudeProvider._FINISH_REASON_MAP for the Claude side).
    # Callers that need the exact vendor value can still read it off
    # `LLMResponse.raw`.
    _FINISH_REASON_MAP = {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "content_filter",
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

        messages = []

        for message in request.messages:
            messages.append(
                {
                    "role": message.role,
                    "content": message.content,
                }
            )

        if all(
            message["role"] == "system"
            for message in messages
        ):
            # ClaudeProvider has always rejected an all-system-message
            # request (it has nothing left to send as a user turn once
            # system messages are extracted). This provider had no
            # equivalent check -- an all-system request would be sent
            # to the OpenAI API as-is and rejected there instead, with
            # whatever error message OpenAI's API happens to return,
            # rather than failing the same clear, provider-level way
            # both providers now share.
            raise ValueError(
                "LLMRequest must contain at least one "
                "non-system message."
            )

        kwargs = {
            "model": request.model,
            "messages": messages,
            "max_tokens": (
                request.max_tokens
                if request.max_tokens is not None
                else self._DEFAULT_MAX_TOKENS
            ),
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

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

        if not isinstance(content, str) or not content:
            # `message.content` is None/empty in two genuinely
            # different situations that used to be indistinguishable
            # (both silently became `content=""`): the model refused
            # the request (the actual text lives in `message.refusal`
            # instead), or the response is empty for some other reason
            # entirely. Surfacing the refusal explicitly, when present,
            # gives the caller an actionable reason instead of a
            # generic empty-response failure discovered several layers
            # later.
            refusal = getattr(message, "refusal", None)

            if isinstance(refusal, str) and refusal:
                raise ValueError(
                    f"OpenAI refused to generate a response: {refusal}"
                )

            raise ValueError(
                "OpenAI response contains no text content."
            )

        return LLMResponse(
            content=content,
            model=getattr(
                response,
                "model",
                request.model,
            ),
            finish_reason=self._normalize_finish_reason(
                getattr(choice, "finish_reason", None)
            ),
            raw=response,
            usage=self._extract_usage(
                getattr(response, "usage", None)
            ),
        )

    @classmethod
    def _normalize_finish_reason(
        cls,
        raw_finish_reason: str | None,
    ) -> str | None:
        if raw_finish_reason is None:
            return None

        return cls._FINISH_REASON_MAP.get(raw_finish_reason, "other")

    @staticmethod
    def _extract_usage(raw_usage) -> TokenUsage | None:
        """
        Build a TokenUsage from OpenAI's own `response.usage`
        (`prompt_tokens`/`completion_tokens`/`total_tokens`, all
        provided directly -- unlike Anthropic's, OpenAI's own response
        already computes the total, so it is trusted here rather than
        recomputed, in case a future usage shape (e.g. reasoning
        tokens) makes prompt+completion not equal total).

        `None` whenever any of the three fields is missing or isn't a
        real int (a mock/test client with no `.usage` attribute at
        all, or a different SDK shape) -- this never fabricates a
        partial or zero usage.
        """

        prompt_tokens = getattr(raw_usage, "prompt_tokens", None)
        completion_tokens = getattr(raw_usage, "completion_tokens", None)
        total_tokens = getattr(raw_usage, "total_tokens", None)

        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (prompt_tokens, completion_tokens, total_tokens)
        ):
            return None

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
