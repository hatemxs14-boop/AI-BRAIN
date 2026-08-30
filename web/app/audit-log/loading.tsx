import { Skeleton } from "@/components/ui/skeleton";

export default function AuditLogLoading() {
  return (
    <main className="mx-auto max-w-6xl space-y-6 p-8">
      <div className="space-y-2">
        <Skeleton className="h-7 w-40" />
        <Skeleton className="h-4 w-72" />
      </div>
      <div className="space-y-2">
        {Array.from({ length: 8 }).map((_, index) => (
          <Skeleton key={index} className="h-12 w-full rounded-md" />
        ))}
      </div>
    </main>
  );
}
