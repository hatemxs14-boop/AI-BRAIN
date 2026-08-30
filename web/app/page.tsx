import { ScrollText, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { BorderBeam } from "@/components/magicui/border-beam";
import { Marquee } from "@/components/magicui/marquee";
import { NumberTicker } from "@/components/magicui/number-ticker";
import { RunTaskForm } from "@/components/run-task-form";
import { getAgents, getRecentAuditEvents, getSystemStatus } from "@/lib/api";
import { iconForAgent, iconForComponent } from "@/lib/icons";

function summarizeAuditEvent(event: Record<string, unknown>): string {
  const name = typeof event.event === "string" ? event.event : "event";
  const parts = [event.subject, event.action, event.resource]
    .filter((part): part is string => typeof part === "string")
    .join(" · ");
  return parts ? `${name}: ${parts}` : name;
}

// A Server Component: fetched once per request, directly from the
// api/ FastAPI backend (Build Phase 33 Part 1) -- no client-side
// loading state needed for this first, mostly-static view (app/
// loading.tsx, Build Phase 34, covers the in-flight state instead).
// The one genuinely interactive piece (submitting a task to the
// Kernel) is its own small Client Component below (RunTaskForm),
// matching Next.js App Router's own "server by default, client only
// where you need interactivity" convention.
export default async function DashboardPage() {
  let status;
  let statusError: string | null = null;
  try {
    status = await getSystemStatus();
  } catch (error) {
    statusError = error instanceof Error ? error.message : String(error);
  }

  let agents;
  let agentsError: string | null = null;
  try {
    agents = await getAgents();
  } catch (error) {
    agentsError = error instanceof Error ? error.message : String(error);
  }

  // Best-effort only: the "Recent activity" strip is a supplementary
  // touch, not a core view (that's the dedicated /audit-log page), so
  // a failure here just hides the strip instead of showing an error
  // box.
  let recentEvents: Record<string, unknown>[] = [];
  try {
    recentEvents = await getRecentAuditEvents(12);
  } catch {
    recentEvents = [];
  }

  const configuredCount = status?.components.filter((c) => c.configured).length ?? 0;
  const totalCount = status?.components.length ?? 0;

  return (
    <main className="mx-auto max-w-6xl space-y-10 p-8">
      <header className="animate-fade-in-up relative overflow-hidden rounded-xl border border-border/60 bg-gradient-to-r from-card to-card/40 px-6 py-5">
        <BorderBeam duration={8} size={90} />
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-violet-500 shadow-lg shadow-primary/30">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">
                Agent kernel control panel
              </h1>
              <p className="text-sm text-muted-foreground">
                Live status, registered agents, and a real task runner over the{" "}
                <code className="text-foreground/80">api/</code> backend.
              </p>
            </div>
          </div>
          {status && (
            <div className="flex gap-6 pr-1">
              <div className="text-right">
                <div className="text-2xl font-semibold tabular-nums">
                  <NumberTicker value={configuredCount} />
                  <span className="text-muted-foreground">/{totalCount}</span>
                </div>
                <p className="text-xs text-muted-foreground">configured</p>
              </div>
              {agents && (
                <div className="text-right">
                  <div className="text-2xl font-semibold tabular-nums">
                    <NumberTicker value={agents.length} />
                  </div>
                  <p className="text-xs text-muted-foreground">agents</p>
                </div>
              )}
            </div>
          )}
        </div>
      </header>

      {recentEvents.length > 0 && (
        <div className="animate-fade-in-up flex items-center gap-3 rounded-lg border border-border/60 bg-card/30">
          <span className="flex shrink-0 items-center gap-1.5 border-r border-border/60 px-3 py-2 text-xs font-medium text-muted-foreground">
            <ScrollText className="h-3.5 w-3.5" />
            Recent activity
          </span>
          <Marquee pauseOnHover className="flex-1 py-0">
            {recentEvents.map((event, index) => (
              <span key={index} className="whitespace-nowrap text-xs text-muted-foreground">
                {summarizeAuditEvent(event)}
              </span>
            ))}
          </Marquee>
        </div>
      )}

      <section>
        <h2 className="mb-3 text-sm font-medium text-muted-foreground">
          System status
        </h2>
        {statusError ? (
          <ErrorNote message={statusError} hint="Is the api/ backend running (uvicorn api.app:create_app --factory)?" />
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {status?.components.map((component, index) => {
              const Icon = iconForComponent(component.name);
              return (
                <Card
                  key={component.name}
                  className="animate-fade-in-up overflow-hidden transition-colors hover:border-primary/40"
                  style={{ animationDelay: `${index * 40}ms` }}
                >
                  <CardHeader className="flex-row items-center gap-2 space-y-0 p-4 pb-2">
                    <Icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <CardTitle className="text-xs">
                      {component.name.replace(/_/g, " ")}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-4 pt-0">
                    <Badge variant={component.configured ? "success" : "outline"}>
                      {component.configured ? "configured" : "not configured"}
                    </Badge>
                    <p className="mt-2 text-xs text-muted-foreground">
                      {component.detail}
                    </p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-3 text-sm font-medium text-muted-foreground">
          Registered agents
        </h2>
        {agentsError ? (
          <ErrorNote message={agentsError} hint="Is the api/ backend running?" />
        ) : (
          <div className="grid gap-3 sm:grid-cols-3">
            {agents?.map((agent, index) => {
              const Icon = iconForAgent(agent.subject);
              return (
                <Card
                  key={agent.subject}
                  className="animate-fade-in-up overflow-hidden transition-colors hover:border-primary/40"
                  style={{ animationDelay: `${index * 60}ms` }}
                >
                  <CardHeader className="flex-row items-center gap-3 space-y-0">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-accent">
                      <Icon className="h-4 w-4 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-base text-foreground">
                        {agent.display_name}
                      </CardTitle>
                      <CardDescription>{agent.description}</CardDescription>
                    </div>
                  </CardHeader>
                </Card>
              );
            })}
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-3 text-sm font-medium text-muted-foreground">
          Run a task
        </h2>
        <Card>
          <CardContent className="p-6">
            <RunTaskForm />
          </CardContent>
        </Card>
      </section>
    </main>
  );
}

function ErrorNote({ message, hint }: { message: string; hint: string }) {
  return (
    <Card className="border-destructive/40 bg-destructive/10">
      <CardContent className="p-4 text-sm">
        <p className="font-medium text-destructive-foreground">{message}</p>
        <p className="mt-1 text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  );
}
