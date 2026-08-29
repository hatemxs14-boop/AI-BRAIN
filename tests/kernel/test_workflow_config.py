"""
Tests for core.kernel.workflow_config: config-driven WorkflowDefinition
construction -- build_task_from_template()/build_workflow_from_config()
(Build Phase 16), and load_workflow_config_file()/load_workflow_
configs_from_directory() (Build Phase 17, loading a workflow config
straight from a JSON file on disk).

Uses the same minimal, isolated fixtures tests/kernel/test_kernel_
workflow.py already established (a bare AgentLoopResult/AgentContext,
no real agent stack) since this module has nothing to do with any
specific agent -- it is a general-purpose config-to-WorkflowDefinition
builder. tests/kernel/test_kernel_workflow_config_integration.py covers
the full real-stack "write_and_review" workflow build_default_kernel()
now wires, both from an inline config dict and from a JSON file on
disk.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agents.agent_context import AgentContext
from core.agents.agent_loop import AgentLoopResult

from core.kernel.kernel import NormalizedTask, WorkflowDefinition, WorkflowStep

from core.kernel.workflow_config import (
    WorkflowConfigError,
    build_task_from_template,
    build_workflow_from_config,
    load_workflow_config_file,
    load_workflow_configs_from_directory,
)


def _loop_result(*, last_result=None) -> AgentLoopResult:
    return AgentLoopResult(
        status="COMPLETED",
        steps=0,
        last_result=last_result,
        reason="test",
        context=AgentContext(task="test"),
    )


# ---------------------------------------------------------------------
# build_task_from_template
# ---------------------------------------------------------------------

def test_build_task_from_template_rejects_empty_subject():
    with pytest.raises(WorkflowConfigError):
        build_task_from_template("", "{original_task}")


def test_build_task_from_template_rejects_empty_template():
    with pytest.raises(WorkflowConfigError):
        build_task_from_template("writer_agent", "   ")


def test_build_task_from_template_rejects_unsupported_placeholder():
    with pytest.raises(WorkflowConfigError):
        build_task_from_template("writer_agent", "Do {something_else}.")


def test_build_task_from_template_original_task_only_needs_no_previous_result():
    build_task = build_task_from_template("writer_agent", "{original_task}")
    assert build_task("Draft a report.", None) == "Draft a report."


def test_build_task_from_template_plain_string_ignores_previous_result():
    build_task = build_task_from_template("writer_agent", "A fixed instruction.")
    assert build_task("Draft a report.", None) == "A fixed instruction."


def test_build_task_from_template_previous_artifact_raises_on_first_step():
    build_task = build_task_from_template(
        "reviewer_agent", "Review {previous_artifact_path}."
    )
    with pytest.raises(WorkflowConfigError):
        build_task("Draft a report.", None)


def test_build_task_from_template_previous_artifact_raises_when_no_artifact():
    build_task = build_task_from_template(
        "reviewer_agent", "Review {previous_artifact_path}."
    )
    previous = _loop_result(last_result=SimpleNamespace(artifacts=()))
    with pytest.raises(WorkflowConfigError):
        build_task("Draft a report.", previous)


def test_build_task_from_template_previous_artifact_substitutes_real_path():
    build_task = build_task_from_template(
        "reviewer_agent", "Review {previous_artifact_path}."
    )
    previous = _loop_result(
        last_result=SimpleNamespace(artifacts=({"path": "report.md"},))
    )
    assert build_task("Draft a report.", previous) == "Review report.md."


def test_build_task_from_template_can_combine_both_placeholders():
    build_task = build_task_from_template(
        "writer_agent",
        "Original: {original_task} -- from: {previous_artifact_path}",
    )
    previous = _loop_result(
        last_result=SimpleNamespace(artifacts=({"path": "finding.md"},))
    )
    assert (
        build_task("Research the topic.", previous)
        == "Original: Research the topic. -- from: finding.md"
    )


# ---------------------------------------------------------------------
# build_workflow_from_config -- structural validation
# ---------------------------------------------------------------------

_VALID_STEPS = (
    {"subject": "writer_agent", "task_template": "{original_task}"},
    {"subject": "reviewer_agent", "task_template": "Review {previous_artifact_path}."},
)


def _valid_config(**overrides):
    config = {
        "name": "write_and_review",
        "description": "Chains writer_agent -> reviewer_agent.",
        "trigger_keywords_all": ("draft", "review"),
        "steps": _VALID_STEPS,
    }
    config.update(overrides)
    return config


def test_build_workflow_from_config_rejects_non_mapping():
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(["not", "a", "mapping"])


def test_build_workflow_from_config_rejects_missing_name():
    config = _valid_config()
    del config["name"]
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(config)


def test_build_workflow_from_config_rejects_empty_description():
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(_valid_config(description=""))


def test_build_workflow_from_config_rejects_missing_trigger_keywords_all():
    config = _valid_config()
    del config["trigger_keywords_all"]
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(config)


def test_build_workflow_from_config_rejects_empty_trigger_keywords_all():
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(_valid_config(trigger_keywords_all=()))


def test_build_workflow_from_config_rejects_a_bare_string_trigger_keywords_all():
    # A bare string is technically a Sequence[str] -- must be rejected
    # explicitly, or every individual character would be (mis)treated
    # as its own one-letter "keyword".
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(_valid_config(trigger_keywords_all="draft"))


def test_build_workflow_from_config_rejects_a_non_string_keyword():
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(
            _valid_config(trigger_keywords_all=("draft", 123))
        )


def test_build_workflow_from_config_rejects_fewer_than_two_steps():
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(_valid_config(steps=(_VALID_STEPS[0],)))


def test_build_workflow_from_config_rejects_a_non_mapping_step():
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(_valid_config(steps=("not-a-mapping", _VALID_STEPS[1])))


def test_build_workflow_from_config_rejects_a_step_missing_subject():
    bad_step = {"task_template": "{original_task}"}
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(_valid_config(steps=(bad_step, _VALID_STEPS[1])))


def test_build_workflow_from_config_rejects_a_step_missing_task_template():
    bad_step = {"subject": "writer_agent"}
    with pytest.raises(WorkflowConfigError):
        build_workflow_from_config(_valid_config(steps=(bad_step, _VALID_STEPS[1])))


# ---------------------------------------------------------------------
# build_workflow_from_config -- successful build
# ---------------------------------------------------------------------

def test_build_workflow_from_config_returns_a_real_workflow_definition():
    workflow = build_workflow_from_config(_valid_config())
    assert isinstance(workflow, WorkflowDefinition)
    assert workflow.name == "write_and_review"
    assert workflow.description == "Chains writer_agent -> reviewer_agent."
    assert [step.subject for step in workflow.steps] == [
        "writer_agent",
        "reviewer_agent",
    ]
    assert all(isinstance(step, WorkflowStep) for step in workflow.steps)


def test_build_workflow_from_config_can_handle_requires_every_keyword():
    workflow = build_workflow_from_config(_valid_config())

    assert workflow.can_handle(
        NormalizedTask(text="Draft a report about the topic, then review it.")
    ) is True

    # "draft" only -- missing "review".
    assert workflow.can_handle(
        NormalizedTask(text="Draft a report about the topic.")
    ) is False

    # "review" only -- missing "draft".
    assert workflow.can_handle(
        NormalizedTask(text="Review the report.")
    ) is False

    # Neither.
    assert workflow.can_handle(
        NormalizedTask(text="Research the topic.")
    ) is False


def test_build_workflow_from_config_can_handle_matches_whole_words_only():
    # "drafted"/"reviewed" must not match "draft"/"review" as
    # substrings -- the same word-boundary convention every hand-
    # written predicate in core/kernel/default_kernel.py already uses
    # (contains_keyword_phrase).
    workflow = build_workflow_from_config(_valid_config())
    assert workflow.can_handle(
        NormalizedTask(text="I already drafted and reviewed this myself.")
    ) is False


def test_build_workflow_from_config_can_handle_is_case_insensitive():
    workflow = build_workflow_from_config(_valid_config())
    assert workflow.can_handle(
        NormalizedTask(text="DRAFT a report and REVIEW it.")
    ) is True


def test_build_workflow_from_config_steps_build_task_end_to_end():
    workflow = build_workflow_from_config(_valid_config())

    step_1, step_2 = workflow.steps

    task_1 = step_1.build_task("Draft a report about the topic, then review it.", None)
    assert task_1 == "Draft a report about the topic, then review it."

    previous = _loop_result(
        last_result=SimpleNamespace(artifacts=({"path": "report.md"},))
    )
    task_2 = step_2.build_task(
        "Draft a report about the topic, then review it.", previous
    )
    assert task_2 == "Review report.md."


# ---------------------------------------------------------------------
# load_workflow_config_file / load_workflow_configs_from_directory
# (Build Phase 17)
# ---------------------------------------------------------------------

def _write_config_file(directory: Path, filename: str, config: dict) -> Path:
    path = directory / filename
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_load_workflow_config_file_raises_when_file_missing():
    with pytest.raises(WorkflowConfigError):
        load_workflow_config_file("/nonexistent/path/does-not-exist.json")


def test_load_workflow_config_file_raises_on_invalid_json():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = tmp_dir / "broken.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(WorkflowConfigError):
            load_workflow_config_file(path)
    finally:
        shutil.rmtree(tmp_dir)


def test_load_workflow_config_file_raises_on_invalid_config_content():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Valid JSON, but missing required fields -- the underlying
        # build_workflow_from_config() error should surface, wrapped
        # with the file's own path.
        path = _write_config_file(tmp_dir, "incomplete.json", {"name": "x"})
        with pytest.raises(WorkflowConfigError):
            load_workflow_config_file(path)
    finally:
        shutil.rmtree(tmp_dir)


def test_load_workflow_config_file_returns_a_real_workflow_definition():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        path = _write_config_file(tmp_dir, "write_and_review.json", _valid_config())
        workflow = load_workflow_config_file(path)
        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "write_and_review"
    finally:
        shutil.rmtree(tmp_dir)


def test_load_workflow_configs_from_directory_returns_empty_for_missing_directory():
    assert load_workflow_configs_from_directory("/nonexistent/workflows-dir") == ()


def test_load_workflow_configs_from_directory_returns_empty_for_empty_directory():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        assert load_workflow_configs_from_directory(tmp_dir) == ()
    finally:
        shutil.rmtree(tmp_dir)


def test_load_workflow_configs_from_directory_ignores_non_json_files():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        (tmp_dir / "README.md").write_text("not a workflow", encoding="utf-8")
        assert load_workflow_configs_from_directory(tmp_dir) == ()
    finally:
        shutil.rmtree(tmp_dir)


def test_load_workflow_configs_from_directory_loads_all_files_in_sorted_order():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        _write_config_file(
            tmp_dir,
            "b_second.json",
            _valid_config(name="second_workflow"),
        )
        _write_config_file(
            tmp_dir,
            "a_first.json",
            _valid_config(name="first_workflow"),
        )

        workflows = load_workflow_configs_from_directory(tmp_dir)

        assert [w.name for w in workflows] == [
            "first_workflow",
            "second_workflow",
        ]
    finally:
        shutil.rmtree(tmp_dir)


def test_load_workflow_configs_from_directory_raises_naming_the_bad_file():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        _write_config_file(tmp_dir, "a_good.json", _valid_config(name="good_one"))
        bad_path = _write_config_file(tmp_dir, "b_bad.json", {"name": "incomplete"})

        with pytest.raises(WorkflowConfigError) as excinfo:
            load_workflow_configs_from_directory(tmp_dir)

        assert str(bad_path) in str(excinfo.value)
    finally:
        shutil.rmtree(tmp_dir)
