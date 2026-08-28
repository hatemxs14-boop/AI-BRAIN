# Research Findings Workspace

This directory is the default approved sandbox root for the
`write_research_findings` tool
(`core/tools/implementations/write_research_findings_tool.py`), wired
to `research_agent` (`core/agents/research_agent.py`).

Unlike `workspace/research_documents/` (read-only reference material
`research_agent` may cite), this directory holds what the agent itself
produces: persisted research findings, per
`core/agents/RESEARCH_AGENT.md`'s Memory Access section ("write
research findings when explicitly authorized").

That "explicitly authorized" requirement is enforced by the security
stack, not left as a convention: this tool is declared `HIGH` risk in
`permissions.json`, so every write attempt returns `APPROVAL_REQUIRED`
until an explicit, attributed approval (`approved=True,
approved_by=...`) is supplied — nothing lands here without a real
approval decision behind it.

The executor rejects any filename that would resolve outside this
root, only writes approved extensions (`.md`, `.json` by default) up
to a size limit (200 KB by default), and never overwrites an existing
file — a revision must use a new filename, keeping the prior finding
intact. See `create_write_research_findings_executor` for the exact
rules.
