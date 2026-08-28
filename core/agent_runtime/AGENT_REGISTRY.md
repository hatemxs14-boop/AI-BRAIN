# AI-BRAIN — AGENT REGISTRY

## Purpose

The Agent Registry is the authoritative catalog of agents available to the
AI-BRAIN Kernel.

The registry describes agent capabilities and boundaries.

It does not contain agent prompts or implementation logic.

---

## Agent Record

Every registered agent should define:

- id
- name
- purpose
- capabilities
- responsibilities
- inputs
- outputs
- tools
- memory_access
- model_requirements
- risk_level
- verification_requirements
- status

---

## Status Values

- proposed
- active
- paused
- deprecated

Only `active` agents may be selected for normal execution.

---

## Selection Rules

The Kernel should select agents based on:

1. Capability match
2. Responsibility match
3. Required tools
4. Risk compatibility
5. Verification requirements
6. Model requirements
7. Current availability

The Kernel should prefer the smallest agent set capable of completing the
task reliably.

---

## Collaboration

Multiple agents may be selected when:

- the task contains independent domains
- independent verification provides meaningful value
- parallel execution reduces total execution time
- specialized expertise is required

Multiple agents must not be used merely to increase complexity.

---

## Boundaries

An agent must not:

- perform responsibilities belonging to another agent without authorization
- bypass tool permissions
- bypass verification
- access memory outside its declared scope
- perform high-risk actions without required approval

---

## Initial Registry

### research_agent

- id: research_agent
- name: Research Agent
- purpose: Conduct structured research and evidence analysis.
- capabilities:
  - research
  - source_analysis
  - evidence_synthesis
  - comparison
  - contradiction_detection
  - uncertainty_assessment
- responsibilities:
  - investigate research objectives
  - gather evidence
  - analyze sources
  - distinguish facts from inference
  - produce structured findings
- inputs:
  - research objective
  - questions
  - constraints
  - approved sources
  - relevant context
- outputs:
  - research_summary
  - findings
  - evidence
  - source_references
  - contradictions
  - assumptions
  - confidence
  - knowledge_gaps
  - recommended_next_actions
- tools:
  - approved read-only search
  - approved document/file reading
- memory_access:
  - approved project memory
  - approved research memory
- model_requirements:
  - multi-step reasoning
  - source analysis
  - structured synthesis
- risk_level: LOW
- verification_requirements:
  - verify important claims
  - identify source conflicts
  - distinguish fact from inference
  - report uncertainty
- status: proposed

### writer_agent

- id: writer_agent
- name: Writer Agent
- purpose: Synthesize already-persisted research findings into a written
  report and publish it when explicitly authorized.
- capabilities:
  - report_drafting
  - evidence_synthesis
  - traceability_verification
- responsibilities:
  - read previously-persisted research findings
  - organize evidence and conclusions into a coherent written structure
  - draft a report addressing a stated objective or audience
  - flag gaps where the available findings do not support a claim
  - publish a finished report when explicitly authorized
- inputs:
  - writing objective
  - research findings to synthesize
  - required output format
  - relevant context
- outputs:
  - published_report
  - findings_used
  - gaps
  - confidence
- tools:
  - approved read-only access to persisted research findings
  - approved report-publishing tool, gated on explicit approval
- memory_access:
  - read approved research findings
  - write approved reports (explicit approval required)
- model_requirements:
  - multi-document synthesis
  - structured long-form writing
  - consistency checking
- risk_level: LOW
- verification_requirements:
  - confirm every claim traces back to a finding actually read
  - flag unsupported claims
  - report uncertainty
- status: active

See core/agents/WRITER_AGENT.md for the full role specification.

### reviewer_agent

- id: reviewer_agent
- name: Reviewer Agent
- purpose: Independently verify an already-published report against the
  research findings it claims to be based on.
- capabilities:
  - independent_verification
  - claim_traceability_checking
  - uncertainty_assessment
- responsibilities:
  - read a previously-published report
  - read the research findings that report claims to draw on
  - compare the report's claims against what the findings actually say
  - identify claims the findings do not support
  - identify findings the report ignored or contradicted
  - report a confidence assessment for the review as a whole
- inputs:
  - the report to review
  - the research findings to check it against
  - relevant context
- outputs:
  - verdict
  - unsupported_claims
  - ignored_findings
  - confidence
- tools:
  - approved read-only access to persisted research findings
  - approved read-only access to published reports
- memory_access:
  - read approved research findings
  - read approved reports
- model_requirements:
  - multi-document comparison
  - claim-by-claim traceability checking
  - uncertainty identification
- risk_level: LOW
- verification_requirements:
  - confirm every flagged-supported claim traces back to a finding actually
    read
  - flag unsupported claims
  - note findings the report never referenced
  - report uncertainty
- status: active

See core/agents/REVIEWER_AGENT.md for the full role specification.

The registry must only contain agents whose responsibilities, interfaces,
tools, memory access, and verification requirements have been explicitly
designed.

---

## Current Status

**research_agent, writer_agent, and reviewer_agent are all real, wired, and
registered with the Kernel** (see core/kernel/default_kernel.py's
`build_default_kernel()` and core/agents/research_agent.py /
core/agents/writer_agent.py / core/agents/reviewer_agent.py) as of Build
Phase 11, completing a research -> write -> review pipeline. This registry
document itself remains the authoritative catalog of declared
capabilities/boundaries; the `status: proposed` value on research_agent's
own entry above is a known, pre-existing documentation staleness (never
updated after Build Phase 1 wired it up) rather than a current gap --
left as-is here rather than silently corrected, per this project's practice
of naming documentation inconsistencies instead of quietly rewriting
history (see the repo baseline doc's Pass 4 section for other examples of
documented-not-fixed spec inconsistencies).