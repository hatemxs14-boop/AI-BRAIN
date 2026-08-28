# Project Memory Workspace

This directory holds the real (v1) Memory Layer's storage file,
`memory.jsonl`, described in `core/memory/MEMORY_SPEC.md` and
implemented by `core/memory/memory_store.py`'s `MemoryStore`.

It backs `research_agent`'s `read_project_memory` tool
(`core/tools/implementations/read_project_memory_tool.py`) --
`core/agents/RESEARCH_AGENT.md`'s Memory Access section has declared
"read approved project memory" as an allowed capability since that
spec was first written; Build Phase 14 is what finally makes it real.

`memory.jsonl` does not need to exist ahead of time -- `MemoryStore`
creates it (and this directory, if it were ever removed) automatically
on first write. An empty or missing file simply means no memory has
been recorded yet; `read_project_memory` returns no results in that
case, not an error.

Every record is append-only and carries its own `verified` flag.
Per `POLICY_SPEC.md`'s Memory Constraints, an unverified record is
recalled, untrusted context -- never a fact to be treated as already
confirmed. See `MEMORY_SPEC.md` for the full trust model, the
secret-rejection rule enforced at write time, and exactly what this
v1 does and does not yet cover.
