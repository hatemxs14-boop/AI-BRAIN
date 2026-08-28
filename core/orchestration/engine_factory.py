from __future__ import annotations

from core.orchestration.orchestration_engine import (
    OrchestrationEngine,
    SequentialOrchestrationEngine,
)


def create_default_orchestration_engine(
    *,
    prefer_langgraph: bool = True,
) -> OrchestrationEngine:
    """
    Build the OrchestrationEngine the Kernel uses when the caller
    doesn't supply one explicitly.

    Tries the real LangGraphOrchestrationEngine first (per
    EXECUTION_ENGINE.md/ARCHITECTURE.md's stated design) when
    `prefer_langgraph` is true. Falls back to
    SequentialOrchestrationEngine only on ImportError -- i.e. only
    when `langgraph` genuinely is not installed -- so a real
    misconfiguration inside an installed langgraph never gets
    silently swallowed here; only its absence does.

    This is the one place in the Kernel/Orchestration layer that
    decides which engine actually runs. Everything else (the Kernel,
    OrchestrationEngine callers) only ever depends on the abstract
    OrchestrationEngine interface.
    """

    if prefer_langgraph:

        try:
            from core.orchestration.langgraph_orchestration_engine import (
                LangGraphOrchestrationEngine,
            )

            return LangGraphOrchestrationEngine()

        except ImportError:
            pass

    return SequentialOrchestrationEngine()
