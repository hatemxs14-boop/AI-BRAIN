"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { AuditEvent } from "@/lib/api";
import { cn } from "@/lib/utils";

// Build Phase 34: the first real UI for the Security Layer's own audit
// log (core/security/engine/audit_logger.py, Build Phase 13) -- reads
// GET /audit-log/recent, an endpoint that already existed from Build
// Phase 33 Part 1 but had no UI consumer until now. Every entry always
// has a real "timestamp" (added by AuditLogger.record() itself) and an
// "event" field naming its type -- confirmed directly against the real
// call sites in core/security/engine/security_decision.py, which emit
// exactly two event kinds: "security_decision" and "execution_outcome"
// (matching this project's own documented "two events per tool call"
// note). Everything else on an entry is event-type-specific, so this
// component shows the common fields as a compact summary row and the
// full JSON payload behind a click -- it never assumes a fixed schema
// beyond "timestamp" + "event", so it stays correct if a future Build
// Phase adds a third event type.
function fieldAsString(event: AuditEvent, key: string): string | null {
  const value = event[key];
  return typeof value === "string" ? value : null;
}

function eventVariant(eventName: string | null): "success" | "destructive" | "outline" {
  if (eventName === "execution_outcome") return "success";
  if (eventName === "security_decision") return "outline";
  return "outline";
}

export function AuditLogTable({ events }: { events: AuditEvent[] }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  function toggle(index: number) {
    setExpanded((previous) => {
      const next = new Set(previous);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }

  if (events.length === 0) {
    return (
      <p className="rounded-lg border border-border p-6 text-center text-sm text-muted-foreground">
        No audit events recorded yet.
      </p>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border border-border">
      {events.map((event, index) => {
        const isOpen = expanded.has(index);
        const eventName = fieldAsString(event, "event");
        const timestamp = fieldAsString(event, "timestamp");
        const subject = fieldAsString(event, "subject");
        const resource = fieldAsString(event, "resource");
        const action = fieldAsString(event, "action");

        return (
          <div
            key={index}
            className={cn(
              "border-b border-border last:border-b-0",
              isOpen && "bg-accent/40"
            )}
          >
            <button
              type="button"
              onClick={() => toggle(index)}
              className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm transition-colors hover:bg-accent/30"
            >
              {isOpen ? (
                <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
              )}
              <Badge variant={eventVariant(eventName)} className="shrink-0">
                {eventName ?? "event"}
              </Badge>
              <span className="min-w-0 flex-1 truncate text-muted-foreground">
                {[subject, action, resource].filter(Boolean).join(" · ") ||
                  "(no subject/action/resource)"}
              </span>
              <span className="shrink-0 font-mono text-xs text-muted-foreground">
                {timestamp ?? "--"}
              </span>
            </button>

            {isOpen && (
              <pre className="overflow-x-auto border-t border-border/60 bg-background/60 px-4 py-3 text-xs text-muted-foreground">
                {JSON.stringify(event, null, 2)}
              </pre>
            )}
          </div>
        );
      })}
    </div>
  );
}
