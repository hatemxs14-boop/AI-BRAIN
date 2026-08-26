# AI-BRAIN — AGENT SPECIFICATION

## Required Fields

Every agent must define:

- id
- name
- purpose
- responsibilities
- inputs
- outputs
- tools
- memory_access
- verification
- error_recovery
- model
- context_budget

## Execution Lifecycle

1. Receive task
2. Load required context
3. Plan
4. Execute
5. Observe result
6. Recover if necessary
7. Verify
8. Return result
9. Persist important context

## Rules

- Minimum necessary tools
- Structured inputs and outputs
- No hidden responsibilities
- No uncontrolled external actions
- Verification required for important results