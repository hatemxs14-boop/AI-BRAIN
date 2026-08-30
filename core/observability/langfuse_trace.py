"""
core/observability/langfuse_trace.py

Build Phase 32: real, self-hosted observability tracing via Langfuse
(github.com/langfuse/langfuse, MIT-licensed except its own `ee/`
enterprise folders) -- item 6, the final item on the 8-item ECC-
informed priority ranking.

What this is: unlike core/agents/guardrails.py (safety) and
core/evaluation/output_quality.py (a CI/dev quality-scoring tool), this
module is LIVE production observability -- it records what actually
happened during a real Kernel.run() (the task, the outcome, the real
token cost) to a real, queryable trace store, so the user can actually
see and search their agents' real production behavior over time,
rather than only the flat, append-only JSON lines
core/security/audit_log.py already writes (Build Phase 13). The two
are complementary, not competing: audit logging is this project's
compliance/security record; Langfuse tracing is its operational
observability layer.

Provider choice: **Langfuse, self-hosted**, chosen after being told it
is genuinely free/MIT-licensed (except a few clearly-marked enterprise
folders this module never touches), self-hostable via Docker Compose
in about five minutes per its own documentation, and that its Python
SDK can be pointed entirely at a self-hosted instance via `base_url` --
never Langfuse's own paid cloud -- directly continuing this project's
now-standing pattern (Build Phases 29 and 31) of choosing self-hosted
infrastructure to keep real cost at zero.

Structural note, matching Build Phase 31's own convention exactly:
`langfuse` is NOT imported at module import time anywhere in this
file. This project's own `TraceRecorder` ABC needs no vendor SDK at
all. `LangfuseTraceRecorder` (below) takes an ALREADY-CONSTRUCTED
vendor client as a plain, duck-typed object -- exactly
core.llm.embeddings.VoyageEmbeddingClient's own "wrap an already-built
vendor client, never isinstance-check it" convention -- so this class
needs no `langfuse` import either. Only
`build_langfuse_trace_recorder_factory()`'s returned factory imports
`langfuse` lazily, at call time, mirroring
core.llm.embeddings.build_embedding_client_factory()'s identical
"never import the vendor SDK until actually needed" shape.

Honest scope, worth being explicit about: this Build Phase wires ONE
trace per `Kernel.run()` call (see core/kernel/kernel.py's own
`trace_recorder` docstring paragraph) -- capturing the task, the final
outcome, and the real aggregate token cost across every attempt and
recovery -- not a step-by-step trace of every individual LLM call made
inside the agent execution loop. A full step-level trace tree would
require threading a trace/span context down through
core/agents/agent_loop.py and core/agents/llm_decision_engine.py,
which this Build Phase deliberately does not attempt: one true,
end-to-end proof at the Kernel level, honestly scoped, beats a
half-verified deep integration across files this specific Build Phase
did not have the room to also verify. `Kernel.run_workflow()` and
`Kernel.resume()` are also NOT yet wired to this -- only `Kernel.run()`
-- an explicit, stated scope limitation, not a silently-assumed one.

Error-handling philosophy, deliberately different from
core.agents.llama_guard.OllamaLlamaGuardClient and core.evaluation.
output_quality.OllamaJudgeModel: those wrap a synchronous `requests`
call that raises on any real failure, because their callers (the
Llama Guard confidence gate, an evaluation script) need to know a call
genuinely failed. Langfuse's own real SDK is documented as doing the
opposite by design -- "cannot break your application: SDK errors are
caught and logged" internally, fully asynchronously, specifically so
tracing calls never introduce a new production failure mode. This
module respects that: `LangfuseTraceRecorder.record_run()` validates
ITS OWN caller's inputs strictly (a project bug, e.g. a non-string
name, is still a real ValueError/TypeError), but does not add a
second, redundant try/except layer around the vendor SDK's own calls
-- and `Kernel.run()`'s own wiring (see kernel.py) wraps the whole
`trace_recorder.record_run()` call in a broad try/except regardless,
on the stronger principle that recording a trace must NEVER be allowed
to prevent a real Kernel.run() result from reaching its caller, no
matter what a vendor SDK does or does not guarantee on its own.

API surface researched directly against Langfuse's own current
documentation (langfuse.com, github.com/langfuse/langfuse) rather than
assumed from training data: the `Langfuse` client's constructor shape
(`public_key`, `secret_key`, `base_url`), the
`start_as_current_observation(as_type="generation", name=..., model=...)`
context-manager API for manual (non-decorator) tracing, and the
`.update(input=..., output=..., metadata=..., usage={"input_tokens":
..., "output_tokens": ..., "total_tokens": ...})` shape for attaching
a real result and real token cost to one observation. `langfuse` is
not installed in this sandbox (no PyPI access, the same situation as
`voyageai` and `deepeval` before it), so this exact shape has never
executed against a real, installed package here -- confirm it resolves
cleanly on first real use, and report the exact error text if not.

Never stores a real API key anywhere in this module -- exactly
core.llm.embeddings.build_embedding_client_factory()'s own standing
rule, applied here to Langfuse's `public_key`/`secret_key` pair.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

from core.llm.token_usage import TokenUsage


class LangfuseConfigError(ValueError):
    """
    Raised when a Langfuse trace-recorder configuration (or the
    environment it references) is missing, malformed, or otherwise
    invalid. Deliberately a ValueError subclass, matching core.llm.
    embeddings.EmbeddingConfigError's own identical convention: this
    always signals a human-fixable configuration mistake -- a missing
    environment variable, an uninstalled package -- never an internal
    bug in this project's own code.
    """


class TraceRecorder(ABC):
    """
    Provider-independent interface for recording one real observability
    trace -- the same one-layer-over role core.llm.llm_client.
    LLMClient, core.llm.embeddings.EmbeddingClient, core.agents.
    llama_guard.LlamaGuardClient, and core.evaluation.output_quality.
    JudgeModel already establish for generation, embeddings, safety
    classification, and quality judging respectively, applied here to
    observability tracing.

    This layer does not:

    - decide agent actions
    - execute tools
    - authorize operations
    - access the Security Layer
    - contain provider-specific business logic
    """

    @abstractmethod
    def record_run(
        self,
        *,
        name: str,
        input_text: str,
        output_text: str,
        status: str,
        metadata: Mapping[str, Any] | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        """
        Record one real trace observation. `name` identifies what kind
        of run this is (e.g. "kernel_run"); `input_text`/`output_text`
        are the real task and real outcome; `status` is this run's
        final status string (e.g. a KernelResult.status value);
        `metadata` is arbitrary additional structured context (e.g.
        the agent subject, recovery attempt count); `usage`, when
        given, is this run's real, aggregate token cost.

        Raises ValueError/TypeError for invalid arguments (a caller
        mistake). Must never raise for a real backend/network failure
        -- see this module's own top-of-file docstring for why that
        responsibility is deliberately NOT this interface's, and is
        instead handled by (a) the real vendor SDK's own documented
        non-raising behavior for LangfuseTraceRecorder, and (b) the
        caller (Kernel.run()) wrapping this call defensively regardless.
        """
        raise NotImplementedError

    def flush(self) -> None:
        """
        Optional hook: block until every trace recorded so far has
        actually been sent. A no-op by default -- only meaningful for
        an implementation (like LangfuseTraceRecorder) backed by an
        asynchronous, batched vendor SDK. Callers running a short-lived
        script (rather than a long-running service) should call this
        once, explicitly, before exiting, so the process does not exit
        before queued traces are actually transmitted -- this module
        deliberately never calls it automatically after every
        record_run(), since doing so would reintroduce the exact
        per-call network latency the real vendor SDK's own async
        design exists to avoid.
        """


class LangfuseTraceRecorder(TraceRecorder):
    """
    Real observability tracing via a self-hosted Langfuse instance,
    wrapping an ALREADY-CONSTRUCTED `langfuse.Langfuse` client --
    mirrors core.llm.embeddings.VoyageEmbeddingClient's own "wrap an
    already-built vendor client, never isinstance-check it" convention
    exactly, so this class itself needs no `langfuse` import at all.
    Use `build_langfuse_trace_recorder_factory()` below to construct
    one from a public-key-env/secret-key-env/base_url triple, the same
    two-step shape core.llm.embeddings.build_embedding_client_factory()
    already established.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def record_run(
        self,
        *,
        name: str,
        input_text: str,
        output_text: str,
        status: str,
        metadata: Mapping[str, Any] | None = None,
        usage: TokenUsage | None = None,
    ) -> None:

        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")

        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("input_text must be a non-empty string.")

        if not isinstance(output_text, str):
            raise TypeError("output_text must be a string.")

        if not isinstance(status, str) or not status.strip():
            raise ValueError("status must be a non-empty string.")

        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None.")

        if usage is not None and not isinstance(usage, TokenUsage):
            raise TypeError("usage must be a TokenUsage or None.")

        full_metadata: dict[str, Any] = (
            dict(metadata) if metadata is not None else {}
        )
        full_metadata["status"] = status

        usage_payload: dict[str, int] | None = None

        if usage is not None:
            usage_payload = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

        with self.client.start_as_current_observation(
            as_type="generation",
            name=name,
        ) as observation:
            observation.update(
                input=input_text,
                output=output_text,
                metadata=full_metadata,
                usage=usage_payload,
            )

    def flush(self) -> None:
        self.client.flush()


def build_langfuse_trace_recorder_factory(
    *,
    public_key_env: str,
    secret_key_env: str,
    base_url: str,
) -> Callable[[], TraceRecorder]:
    """
    Build a zero-argument factory callable returning a fresh
    TraceRecorder -- the exact same lazy-factory shape every other
    factory in this project already uses (core.llm.embeddings.
    build_embedding_client_factory(), core.llm.model_config.
    build_llm_client_factory_from_config(), AgentRegistration.
    build_agent, build_default_kernel()'s own decision_engine_factory).

    Nothing here touches the environment or imports the vendor SDK
    until the returned factory is actually *called*. Calling it:

    1. Reads `public_key_env` and `secret_key_env` from the real
       process environment. Raises LangfuseConfigError immediately if
       either is unset or empty -- before attempting any vendor SDK
       import, mirroring build_embedding_client_factory()'s own
       ordering.
    2. Imports the `langfuse` package and constructs its client with
       those two keys and `base_url`. Raises LangfuseConfigError
       (chained from the original ImportError) with a `pip install`
       instruction if that package is not installed.
    3. Wraps the constructed vendor client in LangfuseTraceRecorder and
       returns it.
    """

    if not isinstance(public_key_env, str) or not public_key_env.strip():
        raise LangfuseConfigError(
            "public_key_env must be a non-empty string."
        )

    if not isinstance(secret_key_env, str) or not secret_key_env.strip():
        raise LangfuseConfigError(
            "secret_key_env must be a non-empty string."
        )

    if not isinstance(base_url, str) or not base_url.strip():
        raise LangfuseConfigError("base_url must be a non-empty string.")

    def factory() -> TraceRecorder:
        import os

        public_key = os.environ.get(public_key_env)

        if not public_key:
            raise LangfuseConfigError(
                f"Environment variable '{public_key_env}' is not set "
                "(or is empty); it must hold the real Langfuse public "
                "key for your self-hosted instance. This module "
                "deliberately never stores the key itself -- only the "
                "name of the environment variable to read it from."
            )

        secret_key = os.environ.get(secret_key_env)

        if not secret_key:
            raise LangfuseConfigError(
                f"Environment variable '{secret_key_env}' is not set "
                "(or is empty); it must hold the real Langfuse secret "
                "key for your self-hosted instance."
            )

        try:
            import langfuse
        except ImportError as exc:
            raise LangfuseConfigError(
                "The 'langfuse' package is not installed, so no real "
                "trace recorder can be built. Install it with `pip "
                "install langfuse`."
            ) from exc

        client = langfuse.Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
        )

        return LangfuseTraceRecorder(client)

    return factory
