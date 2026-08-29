from __future__ import annotations

from pathlib import Path
from typing import Callable

from core.agents.decision_engine import (
    AgentDecisionEngine,
)

from core.agents.llm_decision_engine import (
    LLMDecisionEngine,
)

from core.agents.research_agent import (
    DEFAULT_DOCUMENTS_ROOT,
    DEFAULT_FINDINGS_ROOT,
    DEFAULT_MEMORY_STORE_PATH,
    DEFAULT_PERMISSIONS_PATH,
    build_research_agent,
)

from core.agents.writer_agent import (
    DEFAULT_REPORTS_ROOT,
    build_writer_agent,
)

from core.agents.reviewer_agent import (
    build_reviewer_agent,
)

from core.agents.agent_loop import (
    AgentLoopResult,
)

from core.kernel.kernel import (
    AgentRegistration,
    Kernel,
    NormalizedTask,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowVerifierRegistration,
    contains_keyword_phrase,
    extract_first_artifact_path,
)

from core.kernel.workflow_config import (
    build_workflow_from_config,
    load_workflow_configs_from_directory,
)

from core.llm.caching_llm_client import (
    CachingLLMClient,
    ResponseCache,
)

from core.llm.llm_client import (
    LLMClient,
)

from core.llm.model_config import (
    build_llm_client_factory_from_config,
    load_model_config,
)

from core.memory.memory_store import (
    MemoryStore,
)

from core.orchestration.orchestration_engine import (
    OrchestrationEngine,
)

from core.policies.policy_engine import (
    PolicyEngine,
)


# ---------------------------------------------------------------------
# Convenience wiring: a Kernel with research_agent, writer_agent, and
# (as of Build Phase 11) reviewer_agent all registered, the same way
# core/agents/research_agent.py's own build_research_agent(),
# core/agents/writer_agent.py's own build_writer_agent(), and
# core/agents/reviewer_agent.py's own build_reviewer_agent() are
# convenience wirings of each agent's own tool/security stack.
#
# Until Build Phase 8, research_agent was this project's only
# registered agent, so its `can_handle` was `_always_handles`
# (accepted every task) -- Kernel._classify() had nothing to actually
# classify between. With three agents now registered, all three
# predicates below are real: a finite, hand-maintained keyword
# vocabulary per agent, in the same spirit as RiskEngine's own
# keyword-heuristic classification (Pass 3 finding I) -- not a real
# NLU classifier (no such subsystem exists in this project), but a
# genuine, testable discriminator rather than the previous
# accept-everything placeholder. A task matching more than one
# vocabulary (or none) is handled exactly as
# Kernel._classify()/_select_agent() already document: every match is
# collected, and the first in registration order is selected
# (research_agent first, then writer_agent, then reviewer_agent, so an
# earlier-registered agent wins a genuine tie) -- STRATEGY SELECTION
# remains "run one matching agent", unchanged by this phase; only
# which agents can match at all is new. See Kernel._classify()'s own
# docstring in core/kernel/kernel.py for the unchanged selection
# mechanism itself.
#
# Matching is whole-word, not plain substring: a first draft of this
# module matched keywords with a plain `keyword in text` check, and
# tests/kernel/test_kernel_writer_agent_integration.py's own
# "Summarize finding.md." case caught it immediately -- research_
# agent's "find" keyword is a substring of "finding"/"findings", which
# is exactly the vocabulary writer_agent's own domain (reading
# research *findings*) uses constantly, so plain substring matching
# misrouted a writer_agent task to research_agent on nearly every real
# phrasing. Every keyword below is matched with `\bkeyword\b` instead,
# so "find" no longer matches inside "finding" (no word boundary
# between "find" and the following "ing"), while multi-word phrases
# like "read document" still match as a whole phrase.
#
# Build Phase 11 added a third agent, reviewer_agent, and caught a
# second vocabulary-overlap problem of the same kind before it ever
# shipped: reviewer_agent's whole domain is verifying a *report*, so
# almost any realistic review task ("review the report", "verify the
# report's claims") contains the word "report" -- which
# _WRITER_AGENT_KEYWORDS previously listed as a standalone trigger.
# Since research_agent is registered first, writer_agent second, and
# reviewer_agent third, a task matching both writer_agent's and
# reviewer_agent's vocabulary would have been a "genuine tie" that
# always resolved to writer_agent (registration order), silently
# starving reviewer_agent of almost every realistic phrasing of its
# own job. Checked directly against the test suite before removing
# it: no test relied on the bare word "report" alone triggering
# writer_agent (every existing writer_agent test task already
# contains "draft"/"summarize"/"write" too), so "report" was removed
# from _WRITER_AGENT_KEYWORDS entirely -- writer_agent is now reached
# only by its own drafting/publishing verbs, and reviewer_agent's own
# verification verbs no longer collide with it.
#
# Build Phase 12 gave this Kernel its first real Workflow Constraints
# capability (core/policies/policy_engine.py's evaluate_workflow_
# trigger(), Kernel._trigger_independent_verification): an opt-in
# `enable_independent_verification` flag that, when True, registers
# reviewer_agent as this Kernel's `independent_verifier`, so a
# SUCCESSful writer_agent write_report call automatically triggers a
# real reviewer_agent run afterward. Defaults to False -- CLASSIFY/
# _select_agent's own single-agent-per-task selection mechanism is
# completely unchanged by this; it only adds an optional, purely
# additive extra step after a specific completed action, never a new
# way to select the *primary* agent for a task.
# ---------------------------------------------------------------------


_RESEARCH_AGENT_KEYWORDS: tuple[str, ...] = (
    "research",
    "search",
    "find",
    "investigate",
    "look up",
    "gather",
    "read the document",
    "read document",
    "read the webpage",
    "read webpage",
)

_WRITER_AGENT_KEYWORDS: tuple[str, ...] = (
    "write",
    "draft",
    "summarize",
    "summarise",
    "summary",
    "compose",
)

_REVIEWER_AGENT_KEYWORDS: tuple[str, ...] = (
    "review",
    "verify",
    "audit",
    "validate",
    "critique",
    "fact-check",
    "fact check",
    "double-check",
    "double check",
    "cross-check",
    "cross check",
)


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """
    True if `text` (already lowercased) contains any of `keywords` as
    a whole word/phrase -- `\\bkeyword\\b`, not a plain substring
    check. See this module's own docstring for why plain substring
    matching was wrong (it let research_agent's "find" keyword match
    inside "finding"/"findings").

    Build Phase 16: this is now a thin wrapper around core.kernel.
    kernel.contains_keyword_phrase, which promoted this exact
    word-boundary convention to a shared, module-level helper so
    core/kernel/workflow_config.py's config-driven workflows can reuse
    it too. Kept here, under its original name, so every existing
    call site and docstring reference in this module needs no further
    change.
    """

    return contains_keyword_phrase(text, keywords)


def _research_agent_handles(normalized: NormalizedTask) -> bool:
    """
    Real (v1) capability predicate for research_agent: matches when
    the normalized task text contains any of a finite, hand-maintained
    set of research/evidence-gathering keywords, as a whole word or
    phrase (see _contains_keyword). See this module's own docstring
    for why this is a deliberate keyword heuristic, not a real NLU
    classifier.
    """

    return _contains_keyword(normalized.text.lower(), _RESEARCH_AGENT_KEYWORDS)


def _writer_agent_handles(normalized: NormalizedTask) -> bool:
    """
    Real (v1) capability predicate for writer_agent: matches when the
    normalized task text contains any of a finite, hand-maintained set
    of writing/reporting keywords, as a whole word or phrase (see
    _contains_keyword). See this module's own docstring for why this
    is a deliberate keyword heuristic, not a real NLU classifier.
    """

    return _contains_keyword(normalized.text.lower(), _WRITER_AGENT_KEYWORDS)


def _reviewer_agent_handles(normalized: NormalizedTask) -> bool:
    """
    Real (v1) capability predicate for reviewer_agent: matches when
    the normalized task text contains any of a finite, hand-maintained
    set of independent-verification keywords, as a whole word or
    phrase (see _contains_keyword). See this module's own docstring
    for why this is a deliberate keyword heuristic, not a real NLU
    classifier, and for the "report" vocabulary-overlap problem this
    agent's addition surfaced and fixed on _WRITER_AGENT_KEYWORDS.
    """

    return _contains_keyword(normalized.text.lower(), _REVIEWER_AGENT_KEYWORDS)


# ---------------------------------------------------------------------
# Build Phase 15: this project's first concrete, real
# core.kernel.kernel.WorkflowDefinition -- "research_write_review",
# chaining research_agent -> writer_agent -> reviewer_agent end-to-end
# from a single instruction, via the new Kernel.run_workflow(). See
# WorkflowDefinition's, WorkflowStep's, and Kernel.run_workflow()'s own
# docstrings in core/kernel/kernel.py for the full mechanism and its
# stop-at-approval/stop-at-failure/no-whole-workflow-retry semantics.
# This is deliberately the very same three agents and the very same
# research -> write -> review order AGENT_REGISTRY.md has referenced
# since Build Phase 11 ("completing a research -> write -> review
# pipeline") and Build Phase 12's own `_trigger_independent_verification`
# already automates the LAST hop of (writer_agent -> reviewer_agent,
# after a successful write_report). This workflow does not replace that
# Build Phase 12 mechanism -- a caller can still use plain Kernel.run()
# plus `enable_independent_verification` for the two-stage case -- it
# adds a genuinely new, third entry point for a caller who wants the
# WHOLE three-stage pipeline kicked off from one instruction instead of
# invoking each agent by hand.
#
# `can_handle` is a real, hand-maintained, CONJUNCTIVE predicate, not
# just OR-ing the three agents' own keyword vocabularies together: it
# requires the task to contain a research-signal keyword AND a
# writer-signal keyword (research_write_review's own two starting
# stages -- reviewer_agent's own stage is always implied by choosing
# this workflow, so its vocabulary is deliberately not required here
# too). A plain "write a report" (writer_agent alone, no research
# signal) or "research the topic" (research_agent alone, no write
# signal) does NOT match this workflow's own can_handle -- exactly
# because Kernel.run_workflow() is an entirely separate method from
# Kernel.run() (see run_workflow's own docstring): there is no
# collision to guard against between this workflow's vocabulary and
# any single agent's, since the two are never evaluated by the same
# selection call. The conjunctive design here instead guards against a
# WEAKER mistake this project has already made twice for single-agent
# routing (Build Phase 8's "find"/"finding" substring bug, Build Phase
# 11's "report" keyword-overlap bug): an OR of all three vocabularies
# would let a task that only ever intended a single stage (e.g. "review
# the report", matching only _REVIEWER_AGENT_KEYWORDS) accidentally
# match this workflow too, if a caller ever did decide to try both
# Kernel.run() and Kernel.run_workflow() against the same task text.
# Requiring the research+write signals together keeps this workflow's
# own vocabulary meaningfully distinguishable from any one agent's.
# ---------------------------------------------------------------------


def _research_write_review_handles(normalized: NormalizedTask) -> bool:
    """
    Real (v1) capability predicate for the "research_write_review"
    workflow: matches only when the normalized task text contains BOTH
    a research-signal keyword AND a writer-signal keyword (see this
    module's own docstring above for why conjunctive, not either
    vocabulary alone).
    """

    text = normalized.text.lower()

    return _contains_keyword(
        text, _RESEARCH_AGENT_KEYWORDS
    ) and _contains_keyword(text, _WRITER_AGENT_KEYWORDS)


def _research_write_review_step_1_task(
    original_task: str,
    previous_result: AgentLoopResult | None,
) -> str:
    """
    research_agent's own step: the workflow's original task text,
    verbatim -- there is no previous step to build from yet.
    """

    return original_task


def _research_write_review_step_2_task(
    original_task: str,
    previous_result: AgentLoopResult | None,
) -> str:
    """
    writer_agent's own step: a real task naming the exact findings
    artifact path research_agent's own step just published, built via
    extract_first_artifact_path -- the same convention
    _trigger_independent_verification (Build Phase 12) already
    established for handing one agent's completed artifact to the
    next. Raises ValueError (reported by Kernel.run_workflow() as
    WorkflowRunResult.status == "STEP_TASK_BUILD_ERROR", never as an
    uncaught exception) when `previous_result` is missing or carries no
    usable artifact -- there is nothing for writer_agent to write from
    otherwise.
    """

    if previous_result is None:
        raise ValueError(
            "writer_agent's workflow step requires research_agent's "
            "own completed result from the previous step."
        )

    path = extract_first_artifact_path(previous_result)

    if path is None:
        raise ValueError(
            "research_agent's result carries no usable findings "
            "artifact path for writer_agent to write a report from."
        )

    return f"Write a report summarizing the findings in {path}."


def _research_write_review_step_3_task(
    original_task: str,
    previous_result: AgentLoopResult | None,
) -> str:
    """
    reviewer_agent's own step: a real task naming the exact report
    artifact path writer_agent's own step just published -- the same
    `f"Review {path}."` phrasing
    _trigger_independent_verification (Build Phase 12) already uses for
    this exact hand-off. Raises ValueError (see step 2's own docstring
    for why that is the correct, expected failure mode here) when
    `previous_result` is missing or carries no usable artifact.
    """

    if previous_result is None:
        raise ValueError(
            "reviewer_agent's workflow step requires writer_agent's "
            "own completed result from the previous step."
        )

    path = extract_first_artifact_path(previous_result)

    if path is None:
        raise ValueError(
            "writer_agent's result carries no usable report artifact "
            "path for reviewer_agent to review."
        )

    return f"Review {path}."


def build_default_kernel(
    *,
    llm_client_factory: Callable[[], LLMClient] | None = None,
    decision_engine_factory: Callable[[], AgentDecisionEngine] | None = None,
    documents_root: str | Path = DEFAULT_DOCUMENTS_ROOT,
    findings_root: str | Path = DEFAULT_FINDINGS_ROOT,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    memory_store_path: str | Path = DEFAULT_MEMORY_STORE_PATH,
    serper_api_key: str | None = None,
    permissions_path: str | Path = DEFAULT_PERMISSIONS_PATH,
    audit_log_path: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    orchestration_engine: OrchestrationEngine | None = None,
    max_recovery_attempts: int = 1,
    policy_engine: PolicyEngine | None = None,
    enable_independent_verification: bool = False,
    enable_memory_retrieval: bool = False,
    enable_research_write_review_workflow: bool = False,
    enable_write_and_review_workflow: bool = False,
    workflow_config_dir: str | Path | None = None,
    model_config_path: str | Path | None = None,
    enable_response_cache: bool = False,
    response_cache_max_entries: int = 256,
    response_cache_nondeterministic: bool = False,
) -> Kernel:
    """
    Build a Kernel with research_agent, writer_agent, and
    reviewer_agent all already registered.

    Provide exactly one of `llm_client_factory` (a zero-argument
    callable returning a fresh LLMClient; wrapped in a fresh
    LLMDecisionEngine per Kernel.run() call using `model`/
    `temperature`/`max_tokens`) or `decision_engine_factory` (a
    zero-argument callable returning a fresh AgentDecisionEngine
    directly -- e.g. a DeterministicDecisionEngine for testing, or a
    caller-configured LLMDecisionEngine). The same factory is shared by
    all three agents -- it only encapsulates which model/client is
    used, never per-agent state (each Kernel.run() attempt calls it
    fresh via the selected AgentRegistration's own
    build_decision_engine, per AgentRegistration's own docstring).

    A *factory* is required, not an instance, for the same reason
    AgentRegistration.build_agent is a factory: a decision engine (or
    the LLMClient it wraps) may carry per-run state, and reusing one
    instance across unrelated Kernel.run() calls is a footgun this
    signature avoids by construction.

    `documents_root`/`findings_root`/`serper_api_key` are passed
    straight through to build_research_agent() -- see that function's
    own docstring. `findings_root`/`reports_root` are passed straight
    through to build_writer_agent() -- `findings_root` is deliberately
    the SAME parameter research_agent writes into, so writer_agent
    reads exactly what research_agent has actually persisted (see
    core/agents/writer_agent.py's own docstring for this pipeline
    link). `findings_root`/`reports_root` are likewise passed straight
    through to build_reviewer_agent(), so reviewer_agent reads exactly
    the same findings writer_agent read and exactly what writer_agent
    has actually published. `permissions_path`/`audit_log_path` are
    shared by all three agents' security stacks.

    `policy_engine` is passed straight through to Kernel() -- see its
    own docstring (core/kernel/kernel.py). Defaults to a fresh
    PolicyEngine() when not supplied.

    `enable_independent_verification` (Build Phase 12, default False)
    opts into Kernel's new Workflow Constraints capability: when True,
    a fresh reviewer_agent is registered as this Kernel's
    `independent_verifier` (WorkflowVerifierRegistration), so a
    SUCCESSful writer_agent write_report call automatically triggers a
    real reviewer_agent run afterward, surfaced on
    KernelResult.independent_verification -- see Kernel.__init__'s own
    docstring and core/policies/policy_engine.py's WORKFLOW CONSTRAINTS
    section for exactly what this does and does not cover. Defaults to
    False so this Kernel's behavior, cost (an extra agent/decision-
    engine run, potentially an extra LLM call in production), and
    every existing caller's test counts are completely unchanged unless
    a caller explicitly opts in.

    `memory_store_path` (Build Phase 14) is passed straight through to
    build_research_agent() -- see that function's own docstring --
    regardless of `enable_memory_retrieval` below, since research_agent's
    read_project_memory tool is always wired (it is a real, spec-
    declared, LOW-risk read tool, exactly like web_search/read_document/
    read_webpage; see core/agents/research_agent.py's own docstring).

    `enable_memory_retrieval` (Build Phase 14, default False) opts into
    Kernel's own CONTEXT RETRIEVAL step being real: when True, this
    Kernel is built with a real MemoryStore (backed by
    `memory_store_path`) as its `memory_store`, so every `Kernel.run()`
    call performs a real keyword search and surfaces the result on
    KernelResult.retrieved_context -- see Kernel.__init__'s and
    RetrievedContext's own docstrings (core/kernel/kernel.py) for
    exactly what this is (inspectable-only, never fed back into
    execution) and why it defaults to False for the same "no existing
    caller's behavior or test counts change unless they opt in" reason
    `enable_independent_verification` already established above.

    `enable_research_write_review_workflow` (Build Phase 15, default
    False) opts into this Kernel's first concrete
    core.kernel.kernel.WorkflowDefinition: when True, registers
    "research_write_review" (research_agent -> writer_agent ->
    reviewer_agent, chained via the new Kernel.run_workflow()) using
    fresh research_agent/writer_agent/reviewer_agent factories built
    from this same call's own `documents_root`/`findings_root`/
    `reports_root`/`memory_store_path`/`serper_api_key`/
    `permissions_path`/`audit_log_path` -- exactly the same wiring
    `build_research`/`build_writer`/`build_reviewer` below already use
    for standalone registration, so a task routed through this workflow
    reads and writes the exact same findings/reports locations a
    standalone Kernel.run() call for any of these three agents would.
    Defaults to False for the same "no existing caller's behavior, or
    test counts, change unless they opt in" reason
    `enable_independent_verification` and `enable_memory_retrieval`
    already established above -- registering this workflow does not
    itself change what any ordinary Kernel.run() call does (see
    WorkflowDefinition's own docstring for why the two selection paths
    can never collide), but it is still additive surface area a caller
    must ask for explicitly.

    `enable_write_and_review_workflow` (Build Phase 16, default False)
    opts into a second concrete WorkflowDefinition, "write_and_review"
    (writer_agent -> reviewer_agent, triggered by the task text
    containing both "draft" and "review" as whole words) -- this one
    built from a plain config dict via core.kernel.workflow_config.
    build_workflow_from_config() rather than hand-written can_handle/
    build_task functions, demonstrating that a new workflow can now be
    added as data. Uses the exact same fresh `build_writer`/
    `build_reviewer` factories as standalone registration above, so it
    reads/writes the same `findings_root`/`reports_root` locations.
    Defaults to False for the same reason every other `enable_*` flag
    on this function does.

    `workflow_config_dir` (Build Phase 17, default None) opts into
    loading additional workflows straight from JSON files on disk --
    every "*.json" file directly inside this directory is loaded via
    core.kernel.workflow_config.load_workflow_configs_from_directory()
    and registered on this Kernel, so a NEW workflow can be added by
    writing one JSON file, with no code change to this project at all.
    See that function's own docstring for the exact file format and
    its "fail loud on a bad file" behavior. None (the default) skips
    this entirely -- no directory is read, no behavior changes, same
    "no existing caller's behavior changes unless they opt in" reason
    every other flag on this function already follows.

    `model_config_path` (Build Phase 18, default None) opts into
    building `llm_client_factory` automatically from a single, shared
    core.llm.model_config.ModelConfig JSON file (see that module's own
    docstring, and config/model_config.example.json for the template)
    instead of the caller writing its own vendor-SDK-construction code.
    Only consulted when the caller supplies NEITHER `llm_client_factory`
    NOR `decision_engine_factory` -- an explicit factory of either kind
    always wins, so this is purely an additional way to satisfy the
    "exactly one of the two is required" rule below, never a way to
    loosen or bypass it. When used, this file's own `model`/
    `temperature`/`max_tokens` only fill in whichever of this function's
    own `model`/`temperature`/`max_tokens` parameters were left at their
    default of None -- an explicit `model=`/`temperature=`/`max_tokens=`
    argument to this call always wins over the file. This is the
    concrete mechanism behind the "switch every business project built
    on this Kernel to a new model or provider by editing one shared
    file, not by editing code in each project" requirement -- see
    core/llm/model_config.py's own top-of-file docstring for exactly
    what is and is not verified for real in this project's own sandbox
    (the vendor SDKs themselves are not installed there). None (the
    default) reads no file and changes no existing caller's behavior,
    same reason every other flag on this function already follows.

    `enable_response_cache` (Build Phase 20, default False) opts into
    wrapping the resolved `llm_client_factory` (whether explicitly
    supplied or built from `model_config_path` above) in a
    core.llm.caching_llm_client.CachingLLMClient, backed by ONE shared
    ResponseCache created here and reused by every fresh client
    `decision_engine_factory()` builds -- so a cache hit is possible
    not just within a single Kernel.run() call's own RECOVER IF NEEDED
    retries, but across every separate Kernel.run()/run_workflow() call
    made against this same Kernel instance for as long as it lives.
    Only applies when `decision_engine_factory` itself is NOT
    explicitly supplied by the caller (there is no llm_client_factory
    to wrap in that case) -- a caller that builds its own
    decision_engine_factory can still use CachingLLMClient directly.
    See that class's own docstring for exactly when a response is
    cached (temperature == 0 by default; see
    `response_cache_nondeterministic` below) and what a cache hit does
    to `token_usage` (an explicit, honest zero, never the original
    call's real cost replayed). Defaults to False so every existing
    caller's behavior, cost, and test counts are completely unchanged
    unless explicitly opted in, the same pattern every other `enable_*`
    flag on this function already follows.

    `response_cache_max_entries` (Build Phase 20, default 256) is
    passed straight through to the ResponseCache this creates when
    `enable_response_cache=True` -- ignored otherwise.

    `response_cache_nondeterministic` (Build Phase 20, default False)
    is passed straight through to CachingLLMClient's own
    `cache_nondeterministic` -- ignored when `enable_response_cache`
    is False. See CachingLLMClient's own docstring for what setting
    this to True actually opts into and why it is never the default.
    """

    if decision_engine_factory is None:

        if llm_client_factory is None:

            if model_config_path is not None:
                loaded_model_config = load_model_config(model_config_path)
                llm_client_factory = build_llm_client_factory_from_config(
                    loaded_model_config
                )

                if model is None:
                    model = loaded_model_config.model

                if temperature is None:
                    temperature = loaded_model_config.temperature

                if max_tokens is None:
                    max_tokens = loaded_model_config.max_tokens

            else:
                raise ValueError(
                    "Either llm_client_factory or decision_engine_factory "
                    "must be provided (or model_config_path, to build "
                    "llm_client_factory automatically from a shared "
                    "model/provider config file)."
                )

        if not callable(llm_client_factory):
            raise TypeError(
                "llm_client_factory must be callable."
            )

        if enable_response_cache:
            base_llm_client_factory = llm_client_factory
            shared_response_cache = ResponseCache(
                max_entries=response_cache_max_entries
            )

            def llm_client_factory() -> LLMClient:
                return CachingLLMClient(
                    base_llm_client_factory(),
                    cache=shared_response_cache,
                    cache_nondeterministic=response_cache_nondeterministic,
                )

        def decision_engine_factory() -> AgentDecisionEngine:
            return LLMDecisionEngine(
                llm_client_factory(),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    elif not callable(decision_engine_factory):
        raise TypeError(
            "decision_engine_factory must be callable."
        )

    def build_research():
        return build_research_agent(
            documents_root=documents_root,
            findings_root=findings_root,
            memory_store_path=memory_store_path,
            serper_api_key=serper_api_key,
            permissions_path=permissions_path,
            audit_log_path=audit_log_path,
        )

    def build_writer():
        return build_writer_agent(
            findings_root=findings_root,
            reports_root=reports_root,
            permissions_path=permissions_path,
            audit_log_path=audit_log_path,
        )

    def build_reviewer():
        return build_reviewer_agent(
            findings_root=findings_root,
            reports_root=reports_root,
            permissions_path=permissions_path,
            audit_log_path=audit_log_path,
        )

    independent_verifier = (
        WorkflowVerifierRegistration(
            subject="reviewer_agent",
            build_agent=build_reviewer,
            build_decision_engine=decision_engine_factory,
        )
        if enable_independent_verification
        else None
    )

    memory_store = (
        MemoryStore(str(memory_store_path))
        if enable_memory_retrieval
        else None
    )

    kernel = Kernel(
        orchestration_engine=orchestration_engine,
        max_recovery_attempts=max_recovery_attempts,
        policy_engine=policy_engine,
        independent_verifier=independent_verifier,
        memory_store=memory_store,
    )

    kernel.register_agent(
        AgentRegistration(
            subject="research_agent",
            description=(
                "Conducts structured, read-only research and "
                "persists findings when explicitly approved. See "
                "core/agents/RESEARCH_AGENT.md."
            ),
            can_handle=_research_agent_handles,
            build_agent=build_research,
            build_decision_engine=decision_engine_factory,
        )
    )

    kernel.register_agent(
        AgentRegistration(
            subject="writer_agent",
            description=(
                "Synthesizes already-persisted research findings "
                "into a written report and publishes it when "
                "explicitly approved. See "
                "core/agents/WRITER_AGENT.md."
            ),
            can_handle=_writer_agent_handles,
            build_agent=build_writer,
            build_decision_engine=decision_engine_factory,
        )
    )

    kernel.register_agent(
        AgentRegistration(
            subject="reviewer_agent",
            description=(
                "Independently verifies an already-published report "
                "against the research findings it claims to be based "
                "on. See core/agents/REVIEWER_AGENT.md."
            ),
            can_handle=_reviewer_agent_handles,
            build_agent=build_reviewer,
            build_decision_engine=decision_engine_factory,
        )
    )

    if enable_research_write_review_workflow:
        kernel.register_workflow(
            WorkflowDefinition(
                name="research_write_review",
                description=(
                    "Chains research_agent -> writer_agent -> "
                    "reviewer_agent end-to-end from one instruction. "
                    "See this module's own docstring above."
                ),
                can_handle=_research_write_review_handles,
                steps=(
                    WorkflowStep(
                        subject="research_agent",
                        build_task=_research_write_review_step_1_task,
                    ),
                    WorkflowStep(
                        subject="writer_agent",
                        build_task=_research_write_review_step_2_task,
                    ),
                    WorkflowStep(
                        subject="reviewer_agent",
                        build_task=_research_write_review_step_3_task,
                    ),
                ),
            )
        )

    if enable_write_and_review_workflow:
        kernel.register_workflow(
            build_workflow_from_config(
                {
                    "name": "write_and_review",
                    "description": (
                        "Chains writer_agent -> reviewer_agent "
                        "end-to-end from one instruction -- Build "
                        "Phase 16's config-driven counterpart to "
                        "'research_write_review' above, for a task "
                        "that already has enough context for "
                        "writer_agent to draft directly (no prior "
                        "research_agent step). See core/kernel/"
                        "workflow_config.py's own docstring for the "
                        "config-driven mechanism itself."
                    ),
                    "trigger_keywords_all": ("draft", "review"),
                    "steps": (
                        {
                            "subject": "writer_agent",
                            "task_template": "{original_task}",
                        },
                        {
                            "subject": "reviewer_agent",
                            "task_template": "Review {previous_artifact_path}.",
                        },
                    ),
                }
            )
        )

    if workflow_config_dir is not None:
        for workflow in load_workflow_configs_from_directory(workflow_config_dir):
            kernel.register_workflow(workflow)

    return kernel
