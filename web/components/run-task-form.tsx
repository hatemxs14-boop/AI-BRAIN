"use client";

import { useState } from "react";
import { Loader2, PlayCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ApiError, runKernelTask } from "@/lib/api";
import type { KernelRunResult } from "@/lib/api";

// The one genuinely interactive piece of the dashboard's first slice:
// submit a real task string to POST /kernel/run and show the real
// KernelResult summary that comes back. Deliberately a small, isolated
// Client Component (see app/page.tsx's own comment) -- everything
// else on that page is server-rendered.
export function RunTaskForm() {
  const [task, setTask] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<KernelRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!task.trim() || isRunning) {
      return;
    }

    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const runResult = await runKernelTask(task);
      setResult(runResult);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? `${err.message} (HTTP ${err.status})`
          : err instanceof Error
            ? err.message
            : String(err)
      );
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          value={task}
          onChange={(event) => setTask(event.target.value)}
          placeholder="e.g. Research the latest developments in AI agent frameworks"
          className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          disabled={isRunning}
        />
        <Button type="submit" disabled={isRunning || !task.trim()} className="gap-1.5">
          {isRunning ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <PlayCircle className="h-4 w-4" />
          )}
          {isRunning ? "Running..." : "Run"}
        </Button>
      </form>

      {error && (
        <p className="animate-fade-in-up rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive-foreground">
          {error}
        </p>
      )}

      {result && (
        <div className="animate-fade-in-up space-y-2 rounded-md border border-border p-4">
          <div className="flex items-center gap-2">
            <Badge variant={result.status === "COMPLETED" ? "success" : "outline"}>
              {result.status}
            </Badge>
            {result.subject && (
              <span className="text-sm text-muted-foreground">
                handled by {result.subject}
              </span>
            )}
          </div>
          {result.reason && <p className="text-sm">{result.reason}</p>}
          {result.total_tokens !== null && (
            <p className="text-xs text-muted-foreground">
              {result.total_tokens} tokens ({result.prompt_tokens} prompt +{" "}
              {result.completion_tokens} completion), {result.recovery_attempts}{" "}
              recovery attempt(s)
            </p>
          )}
        </div>
      )}
    </div>
  );
}
