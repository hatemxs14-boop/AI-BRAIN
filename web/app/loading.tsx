import { Skeleton } from "@/components/ui/skeleton";

// Next.js App Router convention: automatically wraps app/page.tsx in a
// <Suspense> boundary and renders this while the Server Component's
// own data fetching (getSystemStatus/getAgents) is in flight -- so a
// slow or unreachable backend shows a recognizable loading shape
// instead of a blank white flash.
export default function DashboardLoading() {
  return (
    <main className="mx-auto max-w-6xl space-y-8 p-8">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-7 w-40" />
          <Skeleton className="h-4 w-64" />
        </div>
        <Skeleton className="h-4 w-32" />
      </div>

      <div className="space-y-3">
        <Skeleton className="h-4 w-32" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} className="h-24 rounded-lg" />
          ))}
        </div>
      </div>

      <div className="space-y-3">
        <Skeleton className="h-4 w-40" />
        <div className="grid gap-3 sm:grid-cols-3">
          {Array.from({ length: 3 }).map((_, index) => (
            <Skeleton key={index} className="h-28 rounded-lg" />
          ))}
        </div>
      </div>

      <div className="space-y-3">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-32 rounded-lg" />
      </div>
    </main>
  );
}
