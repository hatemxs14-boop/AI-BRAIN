import { cn } from "@/lib/utils";

// The standard shadcn/ui Skeleton component, verbatim shape -- a plain
// pulsing div, used by the route-level loading.tsx files below while
// their Server Component siblings await the api/ backend.
function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("animate-pulse rounded-md bg-muted", className)} {...props} />
  );
}

export { Skeleton };
