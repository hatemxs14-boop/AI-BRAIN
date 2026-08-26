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

The registry must only contain agents whose responsibilities, interfaces,
tools, memory access, and verification requirements have been explicitly
designed.

---

## Current Status

DESIGNED — NO ACTIVE AGENTS REGISTERED