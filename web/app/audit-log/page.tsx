import { AuditLogTable } from "@/components/audit-log-table";
import { getRecentAuditEvents } from "@/lib/api";

// Build Phase 34: a real UI for GET /audit-log/recent (Build Phase 33
// Part 1), which had no consumer until now. Server Component, same
// try/catch-then-render pattern as app/page.tsx and
// app/workflows/page.tsx.
export default async function AuditLogPage() {
  let events;
  let error: string | null = null;
  try {
    events = await getRecentAuditEvents(100);
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
  }

  return (
    <main className="mx-auto max-w-6xl space-y-6 p-8">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Audit log</h1>
        <p className="text-sm text-muted-foreground">
          The most recent security-decision and execution-outcome events
          recorded by the Security Layer.
        </p>
      </header>

      {error ? (
        <p className="rounded-md border border-destructive/40 bg-destructive/10 p-4 text-sm">
          {error} -- is the api/ backend running?
        </p>
      ) : (
        <AuditLogTable events={events ?? []} />
      )}
    </main>
  );
}
