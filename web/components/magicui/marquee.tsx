import { type ComponentPropsWithoutRef } from "react";

import { cn } from "@/lib/utils";

// Build Phase 35: hand-ported from Magic UI (MIT-licensed,
// magicui.design/r/marquee.json) -- see border-beam.tsx's own comment
// for the full licensing/adaptation note. One change from the
// published source: Magic UI's `gap-(--gap)` is Tailwind v4's
// CSS-variable-reference shorthand; this project is on Tailwind v3.4,
// whose equivalent bracket syntax is `gap-[var(--gap)]`. The
// `[--duration:40s] [--gap:1rem]` custom-property-setting syntax
// itself is unchanged -- that arbitrary-property form has worked in
// Tailwind since v3.1, it's only the *reference* shorthand that is
// v4-only. The `animate-marquee`/`animate-marquee-vertical` utilities
// and their keyframes are registered in tailwind.config.ts (Build
// Phase 35), the v3 way to add a registry component's custom
// animations (v4's own convention would be a `@theme`/`@keyframes`
// block in globals.css instead).
interface MarqueeProps extends ComponentPropsWithoutRef<"div"> {
  className?: string;
  reverse?: boolean;
  pauseOnHover?: boolean;
  children: React.ReactNode;
  vertical?: boolean;
  repeat?: number;
}

export function Marquee({
  className,
  reverse = false,
  pauseOnHover = false,
  children,
  vertical = false,
  repeat = 4,
  ...props
}: MarqueeProps) {
  return (
    <div
      {...props}
      className={cn(
        "group flex gap-[var(--gap)] overflow-hidden p-2 [--duration:40s] [--gap:1rem]",
        {
          "flex-row": !vertical,
          "flex-col": vertical,
        },
        className
      )}
    >
      {Array(repeat)
        .fill(0)
        .map((_, i) => (
          <div
            key={i}
            className={cn("flex shrink-0 justify-around gap-[var(--gap)]", {
              "animate-marquee flex-row": !vertical,
              "animate-marquee-vertical flex-col": vertical,
              "group-hover:[animation-play-state:paused]": pauseOnHover,
              "[animation-direction:reverse]": reverse,
            })}
          >
            {children}
          </div>
        ))}
    </div>
  );
}
