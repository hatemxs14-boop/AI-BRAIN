"""
Tests for core.agents.checkpoint (Build Phase 22: lightweight,
dependency-free checkpoint/resume for a long-running Kernel task that
survives a PROCESS-level interruption -- see that module's own
docstring for the full design and its honestly-scoped limitations).

This file covers TaskCheckpoint's own validation/serialization and
FileCheckpointStore's save/load/delete mechanics in isolation, plus the
matching LLMDecisionEngine._serialize_tool_result passthrough this
phase added. End-to-end AgentExecutionLoop/Kernel resume behavior is
covered separately in tests/agents/test_agent_loop_checkpoint.py and
tests/kernel/test_kernel_checkpoint_resume.py.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from core.agents.checkpoint import (
    CheckpointStore,
    FileCheckpointStore,
    TaskCheckpoint,
)

from core.agents.llm_decision_engine import LLMDecisionEngine


# ---------------------------------------------------------------------
# TaskCheckpoint validation
# ---------------------------------------------------------------------


def _valid_kwargs(**overrides):
    kwargs = dict(
        checkpoint_id="task-1",
        subject="test_agent",
        task="Research AI agents",
        step_count=2,
        tool_results=(
            {"status": "SUCCESS", "summary": "ok", "artifacts": ["a"]},
        ),
        last_tool_id="web_search",
    )
    kwargs.update(overrides)
    return kwargs


def test_task_checkpoint_accepts_valid_data():
    checkpoint = TaskCheckpoint(**_valid_kwargs())
    assert checkpoint.checkpoint_id == "task-1"
    assert checkpoint.step_count == 2


def test_task_checkpoint_rejects_empty_checkpoint_id():
    with pytest.raises(ValueError):
        TaskCheckpoint(**_valid_kwargs(checkpoint_id="   "))


def test_task_checkpoint_rejects_empty_subject():
    with pytest.raises(ValueError):
        TaskCheckpoint(**_valid_kwargs(subject=""))


def test_task_checkpoint_rejects_empty_task():
    with pytest.raises(ValueError):
        TaskCheckpoint(**_valid_kwargs(task=""))


def test_task_checkpoint_rejects_negative_step_count():
    with pytest.raises(ValueError):
        TaskCheckpoint(**_valid_kwargs(step_count=-1))


def test_task_checkpoint_rejects_non_integer_step_count():
    with pytest.raises(TypeError):
        TaskCheckpoint(**_valid_kwargs(step_count="2"))


def test_task_checkpoint_rejects_bool_step_count():
    with pytest.raises(TypeError):
        TaskCheckpoint(**_valid_kwargs(step_count=True))


def test_task_checkpoint_rejects_non_tuple_tool_results():
    with pytest.raises(TypeError):
        TaskCheckpoint(**_valid_kwargs(tool_results=[{"status": "SUCCESS", "summary": "ok", "artifacts": []}]))


def test_task_checkpoint_rejects_malformed_tool_results_entry():
    with pytest.raises(ValueError):
        TaskCheckpoint(**_valid_kwargs(tool_results=({"status": "SUCCESS"},)))


def test_task_checkpoint_rejects_non_string_last_tool_id():
    with pytest.raises(TypeError):
        TaskCheckpoint(**_valid_kwargs(last_tool_id=123))


def test_task_checkpoint_allows_none_last_tool_id():
    checkpoint = TaskCheckpoint(**_valid_kwargs(last_tool_id=None))
    assert checkpoint.last_tool_id is None


# ---------------------------------------------------------------------
# TaskCheckpoint.to_dict / from_dict round trip
# ---------------------------------------------------------------------


def test_task_checkpoint_round_trips_through_to_dict_and_from_dict():
    original = TaskCheckpoint(**_valid_kwargs())
    restored = TaskCheckpoint.from_dict(original.to_dict())

    assert restored == original


def test_task_checkpoint_from_dict_rejects_missing_fields():
    with pytest.raises(ValueError):
        TaskCheckpoint.from_dict({"checkpoint_id": "task-1"})


def test_task_checkpoint_from_dict_rejects_non_list_tool_results():
    data = TaskCheckpoint(**_valid_kwargs()).to_dict()
    data["tool_results"] = "not-a-list"

    with pytest.raises(TypeError):
        TaskCheckpoint.from_dict(data)


# ---------------------------------------------------------------------
# TaskCheckpoint.from_tool_results -- the projection helper
# ---------------------------------------------------------------------


class _FakeToolExecutionResult:
    """
    Duck-typed stand-in for a real ToolExecutionResult -- deliberately
    NOT the real dataclass (which requires a real SecurityDecision to
    even construct). See checkpoint.py's own module docstring for why
    `_project_tool_result` works off `hasattr`, not `isinstance`.
    """

    def __init__(self, status, summary, artifacts):
        self.status = status
        self.summary = summary
        self.artifacts = artifacts


def test_from_tool_results_projects_a_real_shaped_tool_result():
    fake_result = _FakeToolExecutionResult(
        status="SUCCESS",
        summary="Found it.",
        artifacts=("artifact-1", "artifact-2"),
    )

    checkpoint = TaskCheckpoint.from_tool_results(
        checkpoint_id="task-1",
        subject="test_agent",
        task="Research AI agents",
        step_count=1,
        tool_results=[fake_result],
        last_tool_id="web_search",
    )

    assert checkpoint.tool_results == (
        {
            "status": "SUCCESS",
            "summary": "Found it.",
            "artifacts": ["artifact-1", "artifact-2"],
        },
    )


def test_from_tool_results_passes_through_already_projected_dicts():
    already_projected = {
        "status": "SUCCESS",
        "summary": "From an earlier checkpoint.",
        "artifacts": ["x"],
    }

    checkpoint = TaskCheckpoint.from_tool_results(
        checkpoint_id="task-1",
        subject="test_agent",
        task="Research AI agents",
        step_count=1,
        tool_results=[already_projected],
        last_tool_id=None,
    )

    assert checkpoint.tool_results == (already_projected,)


def test_from_tool_results_handles_none_and_unrecognized_shapes():
    checkpoint = TaskCheckpoint.from_tool_results(
        checkpoint_id="task-1",
        subject="test_agent",
        task="Research AI agents",
        step_count=2,
        tool_results=[None, "a plain string result"],
        last_tool_id=None,
    )

    assert checkpoint.tool_results == (
        {"status": None, "summary": None, "artifacts": []},
        {"status": None, "summary": "a plain string result", "artifacts": []},
    )


def test_from_tool_results_stringifies_a_non_sequence_artifacts_value():
    fake_result = _FakeToolExecutionResult(
        status="SUCCESS",
        summary="ok",
        artifacts=42,
    )

    checkpoint = TaskCheckpoint.from_tool_results(
        checkpoint_id="task-1",
        subject="test_agent",
        task="Research AI agents",
        step_count=1,
        tool_results=[fake_result],
        last_tool_id=None,
    )

    assert checkpoint.tool_results == (
        {"status": "SUCCESS", "summary": "ok", "artifacts": ["42"]},
    )


# ---------------------------------------------------------------------
# LLMDecisionEngine._serialize_tool_result's dict-passthrough addition
# ---------------------------------------------------------------------


def test_serialize_tool_result_passes_through_checkpoint_restored_dicts():
    restored_entry = {
        "status": "SUCCESS",
        "summary": "Restored from a checkpoint.",
        "artifacts": ("a", "b"),
    }

    serialized = LLMDecisionEngine._serialize_tool_result(restored_entry)

    assert serialized == {
        "status": "SUCCESS",
        "summary": "Restored from a checkpoint.",
        "artifacts": ["a", "b"],
    }


def test_serialize_tool_result_still_treats_an_unrelated_dict_as_unrecognized():
    # A dict missing one of the three required keys must NOT be
    # mistaken for an already-projected checkpoint entry -- it still
    # falls through to the pre-existing "unrecognized shape" fallback.
    odd_dict = {"status": "SUCCESS", "summary": "ok"}

    serialized = LLMDecisionEngine._serialize_tool_result(odd_dict)

    assert serialized["status"] is None
    assert serialized["artifacts"] == []


# ---------------------------------------------------------------------
# FileCheckpointStore
# ---------------------------------------------------------------------


@pytest.fixture()
def tmp_dir():
    directory = Path(tempfile.mkdtemp())
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_file_checkpoint_store_is_a_checkpoint_store(tmp_dir):
    store = FileCheckpointStore(tmp_dir)
    assert isinstance(store, CheckpointStore)


def test_file_checkpoint_store_creates_its_own_directory(tmp_dir):
    nested = tmp_dir / "checkpoints" / "nested"
    FileCheckpointStore(nested)
    assert nested.is_dir()


def test_file_checkpoint_store_save_then_load_round_trip(tmp_dir):
    store = FileCheckpointStore(tmp_dir)
    checkpoint = TaskCheckpoint(**_valid_kwargs())

    store.save(checkpoint)
    loaded = store.load("task-1")

    assert loaded == checkpoint


def test_file_checkpoint_store_load_returns_none_when_missing(tmp_dir):
    store = FileCheckpointStore(tmp_dir)
    assert store.load("does-not-exist") is None


def test_file_checkpoint_store_save_overwrites_the_same_id(tmp_dir):
    store = FileCheckpointStore(tmp_dir)

    store.save(TaskCheckpoint(**_valid_kwargs(step_count=1)))
    store.save(TaskCheckpoint(**_valid_kwargs(step_count=5)))

    loaded = store.load("task-1")
    assert loaded.step_count == 5


def test_file_checkpoint_store_delete_removes_the_checkpoint(tmp_dir):
    store = FileCheckpointStore(tmp_dir)
    store.save(TaskCheckpoint(**_valid_kwargs()))

    store.delete("task-1")

    assert store.load("task-1") is None


def test_file_checkpoint_store_delete_is_idempotent(tmp_dir):
    store = FileCheckpointStore(tmp_dir)
    # Deleting a checkpoint that was never saved must not raise.
    store.delete("never-existed")
    store.delete("never-existed")


def test_file_checkpoint_store_exists(tmp_dir):
    store = FileCheckpointStore(tmp_dir)
    assert store.exists("task-1") is False

    store.save(TaskCheckpoint(**_valid_kwargs()))
    assert store.exists("task-1") is True


def test_file_checkpoint_store_rejects_path_traversal_checkpoint_id(tmp_dir):
    store = FileCheckpointStore(tmp_dir)

    with pytest.raises(ValueError):
        store.save(TaskCheckpoint(**_valid_kwargs(checkpoint_id="../evil")))


def test_file_checkpoint_store_rejects_empty_checkpoint_id_on_load(tmp_dir):
    store = FileCheckpointStore(tmp_dir)

    with pytest.raises(ValueError):
        store.load("")


def test_file_checkpoint_store_keeps_different_ids_in_separate_files(tmp_dir):
    store = FileCheckpointStore(tmp_dir)

    store.save(TaskCheckpoint(**_valid_kwargs(checkpoint_id="task-a", step_count=1)))
    store.save(TaskCheckpoint(**_valid_kwargs(checkpoint_id="task-b", step_count=9)))

    assert store.load("task-a").step_count == 1
    assert store.load("task-b").step_count == 9
