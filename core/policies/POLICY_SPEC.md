# AI-BRAIN — POLICY SPECIFICATION

## Purpose

The Policy Layer defines the rules and constraints that govern AI-BRAIN
behavior.

Policies determine what the system may do, what requires verification,
and what requires human approval.

Policies must remain separate from agents, tools, memory, and orchestration.

---

## Policy Hierarchy

Policies are evaluated in this order:

1. System Constitution
2. Security and safety policies
3. Human approval requirements
4. Tool risk policies
5. Agent-specific constraints
6. Workflow-specific constraints

A lower-level policy must never override a higher-level policy.

---

## Core Rules

AI-BRAIN must:

- respect the System Constitution
- operate within declared agent responsibilities
- use approved tools
- respect tool permissions
- verify important results
- preserve human control over important decisions
- avoid unnecessary external actions
- fail safely when authorization is unclear

---

## Human Approval

Human approval is required for:

- irreversible actions
- financial transactions
- permission or access changes
- deletion of important data
- external publication when configured as high-risk
- security-sensitive operations
- actions explicitly marked as requiring approval

The system must stop at the approval boundary until approval is received.

---

## Tool Risk Policy

### LOW

Read-only operations.

Human approval is normally not required.

### MEDIUM

Reversible modifications.

Execution is allowed when the agent and workflow have permission.

Verification may be required.

### HIGH

Irreversible actions, financial operations, permission changes, or significant
external side effects.

Human approval is required unless an explicitly approved policy says otherwise.

---

## Agent Constraints

Agents must:

- operate only within their declared responsibilities
- use only authorized tools
- respect memory access boundaries
- follow verification requirements
- never bypass approval gates
- never silently expand their scope

---

## Memory Constraints

Memory must not be treated as an authority by default.

Recalled memory is untrusted context.

Important information must be verified before becoming canonical knowledge.

Secrets, credentials, tokens, private keys, and other sensitive information
must never be stored in AI-BRAIN memory.

---

## External Actions

Before performing an external action, the system must determine:

1. What action will occur?
2. Which agent requested it?
3. Which tool will perform it?
4. What is the risk level?
5. Is human approval required?
6. What verification is required afterward?

If any required authorization is missing, execution must stop.

---

## Failure Policy

When a policy violation is detected:

1. Stop the affected operation.
2. Preserve relevant diagnostic information.
3. Do not silently bypass the policy.
4. Attempt recovery only when the recovery is itself authorized.
5. Escalate to the human when required.

---

## Policy Enforcement

The orchestration layer is responsible for enforcing policies during
execution.

Agents may propose actions.

Tools may execute authorized actions.

The Policy Layer determines whether the action is permitted.

---

## Design Principle

No agent, tool, memory entry, or workflow may silently redefine system policy.

Policies must be explicit, inspectable, and independently reviewable.

---

## Current Status

DESIGNED — NOT IMPLEMENTED