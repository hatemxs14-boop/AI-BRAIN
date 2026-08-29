from __future__ import annotations

import json
from pathlib import Path
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


# ---------------------------------------------------------------------
# Build Phase 17: loading a workflow config straight from a JSON file
# on disk, and a whole directory of them.
#
# build_workflow_from_config() above (Build Phase 16) turned "write a
# new can_handle function and one build_task function per step" into
# "write a dict" -- a real reduction in what a new workflow costs to
# add, but that dict still had to be written inline in Python (e.g.
# core/kernel/default_kernel.py's own "write_and_review" registration).
# This phase closes the remaining gap: since a workflow config is
# already a plain, JSON-serializable mapping (build_workflow_from_
# config never required anything Python-specific -- strings, lists,
# and dicts only), it can be written as an actual .json file and
# loaded at Kernel-build time with no code change and no import at
# all. This is the most direct answer yet to the user's own standing
# product direction: describing a new commercial workflow now means
# writing one JSON file, not touching this project's source code.
#
# load_workflow_configs_from_directory() fails LOUD -- raising
# WorkflowConfigError immediately, naming the offending file -- on the
# first config file that doesn't load, rather than silently skipping a
# broken file and registering only the rest. This matches this
# project's existing "fail at build time on a bad config, don't
# degrade silently" convention (e.g. PolicyEngine.evaluate_agent_
# scope()/evaluate_agent_permission_alignment() both raise immediately
# on a real mismatch rather than registering a partially-broken agent).
# A caller who genuinely wants best-effort partial loading can still
# call load_workflow_config_file() on each file itself and handle
# WorkflowConfigError per file -- that policy decision is deliberately
# left to the caller rather than baked into this helper.
# ---------------------------------------------------------------------


def load_workflow_config_file(path: str | Path) -> WorkflowDefinition:
    """
    Read one workflow config from a JSON file on disk and build it via
    build_workflow_from_config() -- see that function's own docstring
    for the exact expected shape.

    Raises WorkflowConfigError, naming `path`, when: the file does not
    exist or cannot be read; its contents are not valid JSON; or the
    parsed JSON is not itself a valid workflow config (in which case
    the underlying build_workflow_from_config() error is preserved as
    this error's __cause__ and folded into its message).
    """

    file_path = Path(path)

    if not file_path.is_file():
        raise WorkflowConfigError(
            f"Workflow config file does not exist: {file_path}"
        )

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except OSError as error:
        raise WorkflowConfigError(
            f"Could not read workflow config file {file_path}: {error}"
        ) from error

    try:
        config = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise WorkflowConfigError(
            f"Workflow config file {file_path} is not valid JSON: {error}"
        ) from error

    try:
        return build_workflow_from_config(config)
    except WorkflowConfigError as error:
        raise WorkflowConfigError(
            f"Workflow config file {file_path} is invalid: {error}"
        ) from error


def load_workflow_configs_from_directory(
    directory: str | Path,
) -> tuple[WorkflowDefinition, ...]:
    """
    Load every top-level "*.json" file in `directory` (not recursive)
    as a workflow config, via load_workflow_config_file(), in sorted
    filename order -- a deterministic, reproducible registration order,
    which matters for a genuine can_handle tie (see WorkflowDefinition's
    own docstring: first match in registration order wins).

    Returns an empty tuple when `directory` does not exist, or exists
    but contains no "*.json" files -- a missing or empty workflow
    config directory is not itself an error, mirroring every other
    optional, opt-in config surface in this project (e.g. `enable_*`
    flags on build_default_kernel() that simply do nothing when unset).

    Raises WorkflowConfigError immediately -- naming the offending
    file -- on the first file that fails to load; see this module's
    own docstring above for why this deliberately does not silently
    skip a broken file and register only the rest.
    """

    directory_path = Path(directory)

    if not directory_path.is_dir():
        return ()

    return tuple(
        load_workflow_config_file(config_path)
        for config_path in sorted(directory_path.glob("*.json"))
    )
