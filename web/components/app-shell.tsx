"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Brain, LayoutDashboard, ScrollText, Workflow } from "lucide-react";

import { cn } from "@/lib/utils";

// Build Phase 34: turns the dashboard from two disconnected pages into
// a real application shell -- a persistent left sidebar with the
// AI-BRAIN mark and navigation, matching the Vercel/Linear/Cal.com
// reference aesthetic this project's design brief asked for. Purely
// presentational: it renders `children` (the current route's own
// Server Component) inside the content area, with no data fetching of
// its own.
const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/workflows", label: "Workflow canvas", icon: Workflow },
  { href: "/audit-log", label: "Audit log", icon: ScrollText },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen">
      <aside className="fixed inset-y-0 left-0 z-20 flex w-60 flex-col border-r border-border/80 bg-card/40 backdrop-blur-xl">
        <div className="flex items-center gap-2 px-5 py-5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-violet-500 shadow-lg shadow-primary/30">
            <Brain className="text-white" size={18} />
          </div>
          <div>
            <p className="text-sm font-semibold leading-none tracking-tight">
              AI-BRAIN
            </p>
            <p className="mt-0.5 text-[11px] text-muted-foreground">
              Control panel
            </p>
          </div>
        </div>

        <nav className="flex-1 space-y-1 px-3 py-2">
          {NAV_ITEMS.map((item) => {
            const active =
              item.href === "/" ? pathname === "/" : pathname?.startsWith(item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "group relative flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {active && (
                  <motion.span
                    layoutId="active-nav-pill"
                    className="absolute inset-0 rounded-md bg-accent"
                    transition={{ type: "spring", duration: 0.4, bounce: 0.2 }}
                  />
                )}
                <Icon className="relative z-10 h-4 w-4" />
                <span className="relative z-10">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-border/80 px-5 py-4">
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            A thin client over the <code className="text-foreground/80">api/</code>{" "}
            kernel backend.
          </p>
        </div>
      </aside>

      <div className="pointer-events-none fixed inset-0 z-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_-10%,hsl(var(--primary)/0.18),transparent)]" />

      <main className="relative z-10 ml-60 min-h-screen flex-1">{children}</main>
    </div>
  );
}
