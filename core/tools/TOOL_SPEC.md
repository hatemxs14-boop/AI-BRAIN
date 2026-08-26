# AI-BRAIN — TOOL SPECIFICATION

## Every Tool Must Define

- id
- name
- purpose
- input_schema
- output_schema
- permissions
- risk_level
- error_handling

## Tool Rules

- Tools must have one clear purpose.
- Inputs must be explicit and validated.
- Outputs must be structured.
- High-risk tools require approval.
- Tools must never hide important failures.

## Risk Levels

LOW
- Read-only operations

MEDIUM
- Reversible modifications

HIGH
- Irreversible actions
- External side effects
- Financial or permission changes

## Standard Output

Every tool should return:

- status
- summary
- next_actions
- artifacts