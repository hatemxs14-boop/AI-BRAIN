from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from core.agents.agent_loop import (
    AgentLoopResult,
)

from core.kernel.kernel import (
    NormalizedTask,
    WorkflowDefinition,
    WorkflowStep,
    contains_keyword_phrase,
    extract_first_artifact_path,
)


# ---------------------------------------------------------------------
# Build Phase 16: config-driven core.kernel.kernel.WorkflowDefinition
# construction.
#
# Build Phase 15 gave the project a real, declarative multi-step
# Workflow mechanism (WorkflowDefinition/WorkflowStep/Kernel.
# run_workflow()) -- but registering a new workflow still meant
# writing a new Python function for its `can_handle` predicate and one
# more for every step's `build_task` (see core/kernel/default_kernel.py's
# own `_research_write_review_handles`/`_research_write_review_step_
# *_task`). That is a real barrier to the user's own standing product
# direction (build any profitable commercial workflow/process just by
# describing what's needed): every new workflow currently requires a
# code change and a new test file, not just a description.
#
# This module closes that specific gap -- honestly, not by fabricating
# a natural-language planner (this project still has no such
# subsystem, and pretending otherwise would hide the gap behind code
# that looks like it does more than it does, the same standard every
# prior Build Phase has held to), but by making the two things a
# workflow author actually writes by hand -- "which keywords select
# this workflow" and "what task text does each step run with" -- into
# DATA instead of CODE. A new workflow can now be added by writing one
# dict (see build_workflow_from_config's own docstring for its exact
# shape) -- in Python, in a test, or eventually loaded from a JSON/YAML
# file -- rather than by writing a new can_handle function and one
# build_task function per step.
#
# Two building blocks:
#
#   build_task_from_template()   turns one plain string template (with
#                                 up to two placeholders --
#                                 "{original_task}" and
#                                 "{previous_artifact_path}") into a
#                                 WorkflowStep.build_task callable,
#                                 using exactly the same convention
#                                 core/kernel/default_kernel.py's own
#                                 hand-written Build Phase 15 task
#                                 builders already established (build
#                                 the next step's task text from the
#                                 previous step's own published
#                                 artifact, via extract_first_artifact_
#                                 path; raise when that artifact isn't
#                                 available, which Kernel.run_workflow()
#                                 reports as STEP_TASK_BUILD_ERROR).
#
#   build_workflow_from_config() turns one config dict -- name,
#                                 description, a conjunctive keyword
#                                 trigger list, and an ordered list of
#                                 {subject, task_template} steps -- into
#                                 a complete, ready-to-register
#                                 WorkflowDefinition.
#
# `trigger_keywords_all` is deliberately conjunctive-only (every
# keyword must be present, matched whole-word via contains_keyword_
# phrase) -- the same defensive design core/kernel/default_kernel.py's
# own docstring already explains for "research_write_review"'s
# hand-written can_handle: a config-authored workflow is even more
# likely to be written quickly/by a non-expert than a hand-written
# predicate, so this module does not offer an OR-of-keywords option at
# all in v1 -- a caller who genuinely needs richer trigger logic should
# still write a plain Python can_handle function and build a
# WorkflowDefinition directly, exactly as Build Phase 15 did. Silently
# guessing at a more permissive semantics here would risk exactly the
# kind of accidental over-matching this project has already hit twice
# (see core/kernel/default_kernel.py's own module docstring).
# ---------------------------------------------------------------------


class WorkflowConfigError(ValueError):
    """
    Raised by build_workflow_from_config() (or build_task_from_template())
    when a config dict is malformed. A plain ValueError subclass -- not
    a new taxonomy -- so existing `except ValueError` handling still
    catches it; named separately only so a caller can distinguish "this
    config is malformed" from WorkflowDefinition/WorkflowStep's own
    __post_init__ validation errors (also ValueError) if it wants to.
    """


_ORIGINAL_TASK_FIELD = "original_task"
_PREVIOUS_ARTIFACT_PATH_FIELD = "previous_artifact_path"
_PREVIOUS_ARTIFACT_PATH_PLACEHOLDER = "{" + _PREVIOUS_ARTIFACT_PATH_FIELD + "}"


def build_task_from_template(
    subject: str,
    task_template: str,
) -> Callable[[str, AgentLoopResult | None], str]:
    """
    Build a WorkflowStep.build_task callable from a plain string
    template.

    The template may reference two placeholders via ordinary
    str.format() syntax:

        {original_task}            the workflow's own original
                                    instruction text -- always
                                    available, on every step.
        {previous_artifact_path}   the previous step's first published
                                    artifact path, via this project's
                                    own extract_first_artifact_path
                                    (core/kernel/kernel.py) -- only
                                    available from a step's second
                                    invocation onward (there is no
                                    previous step's result on a
                                    workflow's first step).

    This mirrors exactly what every hand-written Build Phase 15 task
    builder in core/kernel/default_kernel.py already does by hand (e.g.
    `f"Write a report summarizing the findings in {path}."`); this
    function exists so a new workflow step can reuse that same
    convention by writing a template string instead of a new Python
    function.

    Returns a callable matching WorkflowStep.build_task's own
    signature. That callable raises WorkflowConfigError -- reported by
    Kernel.run_workflow() as WorkflowRunResult.status ==
    "STEP_TASK_BUILD_ERROR", never as an uncaught exception, exactly
    like every hand-written Build Phase 15 task builder's own
    ValueError already is -- when the template references
    {previous_artifact_path} but this is the workflow's first step, or
    the previous step's result carries no usable artifact.

    Raises WorkflowConfigError immediately (at build time, not run
    time) if `subject` or `task_template` is not a non-empty string, or
    if `task_template` uses str.format() syntax this function does not
    support (an unknown placeholder, or a positional "{}"/"{0}" field --
    only the two named placeholders above are supported).
    """

    if not isinstance(subject, str) or not subject.strip():
        raise WorkflowConfigError(
            "Each workflow step config must have a non-empty string "
            "'subject'."
        )

    if not isinstance(task_template, str) or not task_template.strip():
        raise WorkflowConfigError(
            f"Workflow step for subject {subject!r} must have a "
            "non-empty string 'task_template'."
        )

    needs_previous_artifact = _PREVIOUS_ARTIFACT_PATH_PLACEHOLDER in task_template

    # Fail fast on an unsupported placeholder/positional field, rather
    # than letting a typo surface later as a confusing STEP_TASK_BUILD_
    # ERROR the first time this step actually runs.
    try:
        task_template.format(
            **{
                _ORIGINAL_TASK_FIELD: "",
                _PREVIOUS_ARTIFACT_PATH_FIELD: "",
            }
        )
    except (KeyError, IndexError) as error:
        raise WorkflowConfigError(
            f"Workflow step for subject {subject!r} has a "
            f"'task_template' using an unsupported placeholder "
            f"({error!r}) -- only {{{_ORIGINAL_TASK_FIELD}}} and "
            f"{{{_PREVIOUS_ARTIFACT_PATH_FIELD}}} are supported."
        ) from error

    def _build_task(
        original_task: str,
        previous_result: AgentLoopResult | None,
    ) -> str:
        substitutions: dict[str, str] = {_ORIGINAL_TASK_FIELD: original_task}

        if needs_previous_artifact:
            if previous_result is None:
                raise WorkflowConfigError(
                    f"Workflow step for subject {subject!r} references "
                    f"{_PREVIOUS_ARTIFACT_PATH_PLACEHOLDER!r}, but this "
                    "is the workflow's first step -- there is no "
                    "previous step's result to build it from."
                )

            path = extract_first_artifact_path(previous_result)

            if path is None:
                raise WorkflowConfigError(
                    f"Workflow step for subject {subject!r} references "
                    f"{_PREVIOUS_ARTIFACT_PATH_PLACEHOLDER!r}, but the "
                    "previous step's result carries no usable artifact "
                    "path."
                )

            substitutions[_PREVIOUS_ARTIFACT_PATH_FIELD] = path

        return task_template.format(**substitutions)

    return _build_task


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowConfigError(
            f"Workflow config field {field_name!r} must be a non-empty "
            "string."
        )
    return value


def build_workflow_from_config(config: Mapping[str, Any]) -> WorkflowDefinition:
    """
    Build a complete WorkflowDefinition from a plain config mapping,
    instead of hand-writing a can_handle function and one build_task
    function per step (see this module's own docstring above for why).

    Expected shape (a plain dict, JSON-serializable):

        {
            "name": "write_and_review",
            "description": "...",
            "trigger_keywords_all": ["draft", "review"],
            "steps": [
                {"subject": "writer_agent", "task_template": "{original_task}"},
                {"subject": "reviewer_agent", "task_template": "Review {previous_artifact_path}."},
            ],
        }

    "name"/"description" are passed straight through to
    WorkflowDefinition -- see its own docstring for their validation.

    "trigger_keywords_all" must be a non-empty sequence of non-empty
    strings. The resulting can_handle predicate matches only when the
    normalized task's lowercased text contains EVERY keyword in this
    list as a whole word/phrase (core.kernel.kernel.contains_keyword_
    phrase) -- conjunctive only; see this module's own docstring above
    for why this deliberately offers no OR-of-keywords option in v1.

    "steps" must be a sequence of at least two mappings, each with a
    "subject" string (naming an agent that must already be registered
    with the Kernel this workflow will be registered on -- Kernel.
    register_workflow() itself enforces that, exactly as it already
    does for a hand-built WorkflowDefinition) and a "task_template"
    string (see build_task_from_template's own docstring for its
    placeholder syntax).

    Raises WorkflowConfigError for any structural problem in `config`
    itself (missing/wrong-typed fields, fewer than two steps, an empty
    trigger_keywords_all). WorkflowDefinition's/WorkflowStep's own
    __post_init__ validation still applies on top of this (e.g. a
    duplicate step subject is not rejected here, since neither
    WorkflowDefinition nor WorkflowStep forbids it either).
    """

    if not isinstance(config, Mapping):
        raise WorkflowConfigError(
            "Workflow config must be a mapping (e.g. a plain dict)."
        )

    name = _require_non_empty_str(config.get("name"), "name")
    description = _require_non_empty_str(config.get("description"), "description")

    trigger_keywords_all = config.get("trigger_keywords_all")

    if (
        not isinstance(trigger_keywords_all, Sequence)
        or isinstance(trigger_keywords_all, (str, bytes))
        or len(trigger_keywords_all) == 0
    ):
        raise WorkflowConfigError(
            "Workflow config field 'trigger_keywords_all' must be a "
            "non-empty list/tuple of non-empty strings."
        )

    keywords: list[str] = []
    for keyword in trigger_keywords_all:
        if not isinstance(keyword, str) or not keyword.strip():
            raise WorkflowConfigError(
                "Workflow config field 'trigger_keywords_all' must "
                "contain only non-empty strings."
            )
        keywords.append(keyword.lower())

    keywords_tuple = tuple(keywords)

    step_configs = config.get("steps")

    if (
        not isinstance(step_configs, Sequence)
        or isinstance(step_configs, (str, bytes))
        or len(step_configs) < 2
    ):
        raise WorkflowConfigError(
            "Workflow config field 'steps' must be a list/tuple of at "
            "least two step configs."
        )

    steps: list[WorkflowStep] = []
    for index, step_config in enumerate(step_configs):
        if not isinstance(step_config, Mapping):
            raise WorkflowConfigError(
                f"Workflow config 'steps'[{index}] must be a mapping "
                "(e.g. a plain dict) with 'subject' and 'task_template'."
            )

        subject = _require_non_empty_str(
            step_config.get("subject"), f"steps[{index}].subject"
        )
        task_template = _require_non_empty_str(
            step_config.get("task_template"), f"steps[{index}].task_template"
        )

        steps.append(
            WorkflowStep(
                subject=subject,
                build_task=build_task_from_template(subject, task_template),
            )
        )

    def _can_handle(normalized: NormalizedTask) -> bool:
        text = normalized.text.lower()
        return all(
            contains_keyword_phrase(text, (keyword,)) for keyword in keywords_tuple
        )

    return WorkflowDefinition(
        name=name,
        description=description,
        can_handle=_can_handle,
        steps=tuple(steps),
    )
