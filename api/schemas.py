"""
api/schemas.py

Build Phase 33 (real UI, Part 1): pydantic request/response models for
the HTTP layer (`api/app.py`). `pydantic` is already a real,
confirmed-installed dependency of this project (see requirements.txt
and every core.llm module) -- unlike `fastapi`, importing it directly
at module level here is exactly as safe as any other core/ module
already does, so this file, like `api/service.py`, is fully
unit-testable in this sandbox without needing `fastapi` installed at
all.

Every model here is a plain data shape only -- no validation logic
beyond what pydantic's own field constraints express declaratively,
and no business logic. Real validation (e.g. "task_text must be
non-empty") lives in `api/service.py`, which every route in
`api/app.py` calls into after pydantic has already parsed the request
body into one of these models.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class KernelRunRequest(BaseModel):
    """Request body for POST /kernel/run."""

    task: str = Field(
        ...,
        min_length=1,
        description="The task text to submit to the Kernel via kernel.run().",
    )


class KernelRunResponseModel(BaseModel):
    """Response body for POST /kernel/run -- mirrors api.service.KernelRunSummary.to_dict()."""

    status: str
    subject: str | None
    reason: str | None
    recovery_attempts: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class ComponentStatusModel(BaseModel):
    """Mirrors api.service.ComponentStatus."""

    name: str
    configured: bool
    detail: str


class SystemStatusModel(BaseModel):
    """Mirrors api.service.SystemStatus."""

    components: list[ComponentStatusModel]
    all_configured: bool


class AgentSummaryModel(BaseModel):
    """Mirrors api.service.AgentSummary."""

    subject: str
    display_name: str
    description: str


class HealthResponseModel(BaseModel):
    """Response body for GET /health."""

    status: str
