"use client";

import { motion, type Transition } from "framer-motion";

import { cn } from "@/lib/utils";

// Build Phase 35: hand-ported from Magic UI (magicuidesign/magicui,
// MIT-licensed, magicui.design/r/border-beam.json) -- verified MIT
// before use, same due-diligence standard this project applied to
// ComfyUI/Flowise/n8n/React-Flow. NOT copy-pasted verbatim: Magic UI's
// published source targets Tailwind CSS v4 (`bg-linear-to-l`,
// `from-(--x)`, `mask-[...]`/`mask-intersect`, `border-(length:--x)`)
// and the newer `motion/react` package -- this project is on Tailwind
// v3.4 and `framer-motion` (added in Build Phase 34), neither of which
// understands that v4-only syntax. Rewritten below with the exact same
// visual technique (an animated gradient segment traveling around the
// border via CSS `offset-path`, masked to a border-only ring via the
// standard `mask-composite: exclude` trick) expressed as plain inline
// styles instead of Tailwind utility classes, so it doesn't depend on
// which Tailwind major version is installed.
interface BorderBeamProps {
  size?: number;
  duration?: number;
  delay?: number;
  colorFrom?: string;
  colorTo?: string;
  transition?: Transition;
  className?: string;
  reverse?: boolean;
  initialOffset?: number;
  borderWidth?: number;
}

export function BorderBeam({
  className,
  size = 60,
  delay = 0,
  duration = 6,
  colorFrom = "hsl(217 91% 60%)",
  colorTo = "hsl(270 91% 65%)",
  transition,
  reverse = false,
  initialOffset = 0,
  borderWidth = 1,
}: BorderBeamProps) {
  return (
    <div
      className="pointer-events-none absolute inset-0 rounded-[inherit]"
      style={{
        padding: borderWidth,
        WebkitMask:
          "linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0)",
        WebkitMaskComposite: "xor",
        maskComposite: "exclude",
      }}
    >
      <motion.div
        className={cn("absolute aspect-square", className)}
        style={{
          width: size,
          offsetPath: `rect(0 auto auto 0 round ${size}px)`,
          background: `linear-gradient(to left, ${colorFrom}, ${colorTo}, transparent)`,
        }}
        initial={{ offsetDistance: `${initialOffset}%` }}
        animate={{
          offsetDistance: reverse
            ? [`${100 - initialOffset}%`, `${-initialOffset}%`]
            : [`${initialOffset}%`, `${100 + initialOffset}%`],
        }}
        transition={{
          repeat: Infinity,
          ease: "linear",
          duration,
          delay: -delay,
          ...transition,
        }}
      />
    </div>
  );
}
