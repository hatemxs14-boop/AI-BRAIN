"""
api/app.py

Build Phase 33 (real UI, Part 1): the actual HTTP layer -- a thin
FastAPI wrapper around `api/service.py`'s pure business logic.

This is the FIRST module in this project that imports `fastapi` at
module level. Every previous "optional vendor package" this project
has wired (voyageai, deepeval, langfuse) lazily imports its SDK only
inside a factory's returned callable, specifically so the REST of this
project's test suite never depends on that package being installed.
`fastapi` cannot reasonably follow that same pattern here: it is not
an optional plug-in swapped for a config value, it is the actual web
framework this entire module is built on -- there is no sensible way
to "lazily import a web framework" function-by-function inside every
single route. So this module does a normal, top-level
`from fastapi import ...`, exactly like `core/llm/providers/*.py`
already do a normal, top-level `import anthropic`/`import openai` for
THEIR real, required vendor SDKs.

`fastapi` is not installed in this sandbox (confirmed: no PyPI access
here, the same situation as voyageai/deepeval/langfuse before it), so
this module cannot be imported here either. `tests/api/test_app.py`
guards its own WHOLE file behind `pytest.importorskip("fastapi")` at
the top -- mirroring exactly how tests/llm/test_model_config.py
already treats `anthropic`/`openai` -- rather than trying to lazily
import fastapi inside this module. `api/service.py` (the actual
business logic) has ZERO dependency on fastapi and is fully
unit-tested in this sandbox regardless; see that module's own
docstring.

Every route below is a thin translation: parse the request -> call
exactly one `api/service.py` function -> serialize its result. No
business logic lives in this file.
"""
from __future__ import annotations

import os
from typing import Callable

from fastapi import FastAPI, HTTPException

from api.schemas import (
    AgentSummaryModel,
    ComponentStatusModel,
    HealthResponseModel,
    KernelRunRequest,
    KernelRunResponseModel,
    SystemStatusModel,
)
from api.service import (
    get_system_status,
    list_agents,
    read_recent_audit_events,
    run_kernel_task,
)
from core.kernel.default_kernel import build_default_kernel
from core.kernel.kernel import Kernel


DEFAULT_AUDIT_LOG_PATH_ENV = "AI_BRAIN_AUDIT_LOG_PATH"
DEFAULT_AUDIT_LOG_PATH = "logs/audit.jsonl"

DEFAULT_MODEL_CONFIG_PATH_ENV = "AI_BRAIN_MODEL_CONFIG_PATH"
DEFAULT_MODEL_CONFIG_PATH = "config/model_config.json"


def create_app(
    *,
    kernel_factory: Callable[[], Kernel] | None = None,
    audit_log_path: str | None = None,
) -> FastAPI:
    """
    Build a FastAPI app exposing this project's real Kernel over HTTP.

    `kernel_factory`, when given, is called ONCE here to build the
    single Kernel instance this app's whole process lifetime serves --
    reusing one Kernel across many requests is correct and safe:
    `Kernel.run()` already builds a fresh agent/decision-engine
    internally on every single call (see `AgentRegistration`'s own
    docstring), so the Kernel object itself carries no per-request
    state that a second concurrent request could corrupt. Tests always
    pass their own `kernel_factory` (a stub), exactly like every other
    test file in this project already does for `build_default_kernel()`
    itself.

    When `kernel_factory` is None (the real production path), a real
    Kernel is built via `build_default_kernel()`, reading the same
    environment variables this project's own PRODUCTION ACTIVATION
    CHECKLIST documents: `SERPER_API_KEY` for research_agent's web
    search tool, and an LLM provider key indirectly through
    `model_config_path` (`AI_BRAIN_MODEL_CONFIG_PATH`, defaulting to
    `config/model_config.json` -- see `core/llm/model_config.py`'s own
    `load_model_config`/`build_llm_client_factory_from_config`).

    Deliberately does NOT let a failure to build that real Kernel
    (e.g. a missing `SERPER_API_KEY`, or `config/model_config.json`
    not existing yet) crash the whole app at startup: this project's
    own standing constraint is that the system must never become so
    strict it refuses to run at all. Instead, `app.state.kernel` is
    `None` and `app.state.kernel_build_error` holds the real error
    message; `GET /health` and `GET /system/status` still work fully
    either way (so the dashboard can always show the user exactly
    what is and isn't configured), and only `POST /kernel/run` reports
    a real `503` naming the underlying error when it is actually
    called with no working Kernel.
    """
    resolved_audit_log_path = audit_log_path or os.environ.get(
        DEFAULT_AUDIT_LOG_PATH_ENV, DEFAULT_AUDIT_LOG_PATH
    )

    app = FastAPI(title="AI-BRAIN API", version="0.1.0")

    kernel: Kernel | None
    kernel_build_error: str | None

    if kernel_factory is not None:
        kernel = kernel_factory()
        kernel_build_error = None
    else:
        try:
            kernel = build_default_kernel(
                model_config_path=os.environ.get(
                    DEFAULT_MODEL_CONFIG_PATH_ENV,
                    DEFAULT_MODEL_CONFIG_PATH,
                ),
                serper_api_key=os.environ.get("SERPER_API_KEY"),
                audit_log_path=resolved_audit_log_path,
            )
            kernel_build_error = None
        except Exception as exc:  # noqa: BLE001 -- see docstring above
            kernel = None
            kernel_build_error = str(exc)

    app.state.kernel = kernel
    app.state.kernel_build_error = kernel_build_error
    app.state.audit_log_path = resolved_audit_log_path

    @app.get("/health", response_model=HealthResponseModel)
    def health() -> HealthResponseModel:
        return HealthResponseModel(status="ok")

    @app.get("/system/status", response_model=SystemStatusModel)
    def system_status() -> SystemStatusModel:
        status = get_system_status()
        return SystemStatusModel(
            components=[
                ComponentStatusModel(
                    name=component.name,
                    configured=component.configured,
                    detail=component.detail,
                )
                for component in status.components
            ],
            all_configured=status.all_configured,
        )

    @app.get("/agents", response_model=list[AgentSummaryModel])
    def agents() -> list[AgentSummaryModel]:
        return [
            AgentSummaryModel(
                subject=agent.subject,
                display_name=agent.display_name,
                description=agent.description,
            )
            for agent in list_agents()
        ]

    @app.post("/kernel/run", response_model=KernelRunResponseModel)
    def kernel_run(request: KernelRunRequest) -> KernelRunResponseModel:
        if app.state.kernel is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The Kernel is not available: "
                    f"{app.state.kernel_build_error}. Check this "
                    "project's PRODUCTION ACTIVATION CHECKLIST -- "
                    "likely a missing SERPER_API_KEY or "
                    "config/model_config.json."
                ),
            )

        try:
            summary = run_kernel_task(app.state.kernel, request.task)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return KernelRunResponseModel(**summary.to_dict())

    @app.get("/audit-log/recent")
    def audit_log_recent(limit: int = 50) -> list[dict]:
        try:
            return read_recent_audit_events(
                app.state.audit_log_path, limit=limit
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
