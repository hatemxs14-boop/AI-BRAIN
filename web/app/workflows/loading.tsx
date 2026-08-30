import { Skeleton } from "@/components/ui/skeleton";

export default function WorkflowsLoading() {
  return (
    <main className="mx-auto max-w-6xl space-y-6 p-8">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-7 w-48" />
          <Skeleton className="h-4 w-72" />
        </div>
        <Skeleton className="h-4 w-24" />
      </div>
      <Skeleton className="h-[600px] w-full rounded-lg" />
    </main>
  );
}
