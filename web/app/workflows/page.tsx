import { WorkflowCanvas } from "@/components/workflow-canvas";
import { getAgents } from "@/lib/api";

// Build Phase 34: the "<- Dashboard" back-link was removed here --
// navigation between routes is now handled by the persistent sidebar
// in components/app-shell.tsx (wrapping every page via app/layout.tsx),
// so a second, page-local nav link would just be a duplicate.
export default async function WorkflowsPage() {
  let agents;
  let error: string | null = null;
  try {
    agents = await getAgents();
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
  }

  return (
    <main className="mx-auto max-w-6xl space-y-6 p-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">
          Workflow canvas
        </h1>
        <p className="text-sm text-muted-foreground">
          The registered agents, in their documented pipeline order.
        </p>
      </header>

      {error ? (
        <p className="rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm">
          {error} -- is the api/ backend running?
        </p>
      ) : (
        <WorkflowCanvas agents={agents ?? []} />
      )}
    </main>
  );
}
