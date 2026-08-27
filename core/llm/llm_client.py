from __future__ import annotations

from abc import ABC, abstractmethod

from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse


class LLMClient(ABC):
    """
    Provider-independent interface for communicating with an LLM.

    Concrete providers must implement this interface.

    This layer does not:

    - decide agent actions
    - execute tools
    - authorize operations
    - access the Security Layer
    - contain provider-specific business logic
    """

    @abstractmethod
    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate an LLM response from a normalized request.
        """
        raise NotImplementedError
