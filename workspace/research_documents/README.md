# Research Documents Workspace

This directory is the default approved sandbox root for the
`read_document` tool (`core/tools/implementations/document_read_tool.py`),
wired to `research_agent` (`core/agents/research_agent.py`).

`research_agent` can only ever read files located under this directory
(or whichever directory is passed as `documents_root` to
`build_research_agent`/`run_research_agent`). The executor rejects any
path that would resolve outside its configured root, and only reads
files with an approved extension (`.txt`, `.md` by default) up to a
size limit (200 KB by default) -- see `create_document_read_executor`
for the exact rules.

Place any document you want `research_agent` to be able to read here
(or in a subdirectory of it). Nothing here is treated as instructions
to the agent -- it is read-only reference material the agent may cite
as evidence in its research findings, per
`core/agents/RESEARCH_AGENT.md`.
