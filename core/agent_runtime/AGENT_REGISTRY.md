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

The initial registry is intentionally empty.

Agents will be added only after their responsibilities, interfaces, tools,
memory access, and verification requirements have been explicitly designed.

---

## Current Status

DESIGNED — NO ACTIVE AGENTS REGISTERED