# AI-BRAIN — SYSTEM CONSTITUTION

## 1. Purpose

AI-BRAIN is a modular AI system designed to reason, remember, plan, execute, verify, and improve.

## 2. Core Principles

- Modular architecture
- Clear separation of responsibilities
- Persistent and organized memory
- Easy modification and expansion
- Human control over important decisions
- Verification before important actions
- No unnecessary complexity

## 3. Main Components

- Claude: reasoning and intelligence
- LangGraph: orchestration and state management
- ECC: agents, skills, rules, and engineering patterns
- Obsidian: long-term knowledge and memory
- n8n: external automation and integrations
- UI: visual control and management interface

## 4. Architecture Rule

No component should unnecessarily control the responsibilities of another component.

## 5. Development Rule

The system must be designed and tested before expensive AI services are activated.

## 6. Human Control

The human remains the final decision-maker for important actions.

## 7. Modularity

Agents, skills, tools, memory, workflows, and integrations must be independently replaceable whenever practical.

## 8. Current Status

BUILDING — NOT OPERATIONAL

## Agent Design Standard

Every agent must have:

- A clearly defined responsibility
- A limited and explicit toolset
- Structured tool inputs
- Structured outputs
- Error recovery behavior
- Context budget controls
- Verification before completion

Recommended architecture:

Hybrid — reasoning/planning + typed tool execution.

Agents must not have overlapping responsibilities without a defined reason.