import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

// The standard shadcn/ui `cn()` helper, verbatim -- merges Tailwind
// classes safely (later classes win over conflicting earlier ones),
// used by every component under components/ui/.
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
