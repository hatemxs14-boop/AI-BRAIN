// Build Phase 34: a small, explicit mapping from domain concepts (agent
// subjects, system-status component names) to a lucide-react icon
// component. Kept in one place so app/page.tsx and workflow-canvas.tsx
// don't each invent their own guessing logic -- unknown names always
// fall back to a sensible default rather than throwing.

import {
  Bot,
  Brain,
  Cpu,
  Database,
  FileSearch,
  Gauge,
  KeyRound,
  Layers,
  type LucideIcon,
  PenLine,
  Search,
  Server,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

const AGENT_ICONS: Record<string, LucideIcon> = {
  research_agent: FileSearch,
  writer_agent: PenLine,
  reviewer_agent: ShieldCheck,
};

export function iconForAgent(subject: string): LucideIcon {
  return AGENT_ICONS[subject] ?? Bot;
}

// Component-status names come from api/service.py's get_system_status()
// (e.g. "llm_provider", "serper_api_key", "voyage_api_key",
// "ollama_llama_guard", "deepeval", "langfuse"). Matched by substring so
// this stays correct even if the exact set of components grows.
const COMPONENT_ICON_RULES: Array<[pattern: RegExp, icon: LucideIcon]> = [
  [/llm|anthropic|openai|model/i, Brain],
  [/serper|search/i, Search],
  [/voyage|embed/i, Layers],
  [/guard|safety/i, ShieldCheck],
  [/deepeval|quality|eval/i, Gauge],
  [/langfuse|trace|observ/i, Sparkles],
  [/key|secret|token/i, KeyRound],
  [/database|store|memory/i, Database],
];

export function iconForComponent(name: string): LucideIcon {
  for (const [pattern, icon] of COMPONENT_ICON_RULES) {
    if (pattern.test(name)) {
      return icon;
    }
  }
  return Cpu;
}

export { Server, Sparkles };
