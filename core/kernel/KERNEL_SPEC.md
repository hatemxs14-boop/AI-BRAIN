# AI-BRAIN — KERNEL SPECIFICATION

## 1. Purpose

The Brain Kernel is the central control layer of AI-BRAIN.

Its responsibility is to transform a user objective into a controlled,
verifiable execution process.

The Kernel does not perform specialized work itself.

It coordinates reasoning, planning, agents, tools, memory, verification,
human approval, and learning.

---

## 2. Core Responsibilities

The Kernel must:

1. Receive the objective.
2. Normalize and classify the task.
3. Determine required context.
4. Retrieve relevant memory.
5. Determine execution strategy.
6. Select appropriate agent(s).
7. Select appropriate model(s).
8. Select required tools.
9. Construct an execution plan.
10. Execute through the orchestration layer.
11. Observe execution results.
12. Recover from failures.
13. Verify important results.
14. Request human approval when required.
15. Persist approved durable knowledge.
16. Evaluate the completed task.
17. Record useful lessons.
18. Return a structured final result.

---

## 3. Non-Responsibilities

The Kernel must not:

- perform specialized domain work when an agent exists for it
- directly execute external side effects
- bypass tool permissions
- bypass verification
- treat memory as authoritative without verification
- store secrets
- silently make irreversible decisions
- replace the orchestration layer
- replace specialized agents
- replace the memory layer

---

## 4. High-Level Execution Lifecycle

```text
INPUT
  ↓
NORMALIZE
  ↓
CLASSIFY
  ↓
CONTEXT RETRIEVAL
  ↓
STRATEGY SELECTION
  ↓
AGENT SELECTION
  ↓
MODEL SELECTION
  ↓
TOOL SELECTION
  ↓
PLAN
  ↓
EXECUTE
  ↓
OBSERVE
  ↓
RECOVER IF NEEDED
  ↓
VERIFY
  ↓
HUMAN APPROVAL IF REQUIRED
  ↓
PERSIST
  ↓
EVALUATE
  ↓
LEARN
  ↓
FINAL RESULT