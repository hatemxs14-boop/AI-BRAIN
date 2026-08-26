# AI-BRAIN — SECURITY SPECIFICATION

## Purpose

The Security Layer protects AI-BRAIN from unauthorized actions, privilege
escalation, unsafe tool usage, and uncontrolled external effects.

Security is independent from agent reasoning and workflow orchestration.

---

## Security Principles

AI-BRAIN follows these principles:

- Least privilege
- Explicit permissions
- Default deny
- Separation of responsibilities
- Human control for high-risk operations
- No silent privilege escalation
- Complete and inspectable authorization decisions
- Fail closed when authorization is uncertain

---

## Permission Model

Every executable capability must have an explicit permission definition.

A permission should identify:

- subject
- resource
- action
- scope
- risk_level
- approval_requirement

Example:

```text
subject: research_agent
resource: web_search
action: read
scope: public_web
risk_level: LOW
approval: none