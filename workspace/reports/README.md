# Reports Workspace

This directory is the default approved sandbox root for the
`write_report` tool
(`core/tools/implementations/write_report_tool.py`), wired to
`writer_agent` (`core/agents/writer_agent.py`).

`writer_agent` reads persisted research findings from
`workspace/research_findings/` (via its own `read_research_findings`
tool) and synthesizes them into a written report, which lands here —
per `core/agents/WRITER_AGENT.md`'s Memory Access section ("publish a
written report when explicitly authorized").

That "explicitly authorized" requirement is enforced by the security
stack, not left as a convention: this tool is declared `HIGH` risk in
`permissions.json`, so every write attempt returns `APPROVAL_REQUIRED`
until an explicit, attributed approval (`approved=True,
approved_by=...`) is supplied — nothing lands here without a real
approval decision behind it.

The executor rejects any filename that would resolve outside this
root, only writes approved extensions (`.md` by default) up to a size
limit (200 KB by default), and never overwrites an existing file — a
revision must use a new filename, keeping the prior report intact. See
`create_write_report_executor` for the exact rules.
