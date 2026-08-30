import type { Metadata } from "next";
import "./globals.css";

import { AppShell } from "@/components/app-shell";

export const metadata: Metadata = {
  title: "AI-BRAIN",
  description: "Control panel for the AI-BRAIN agent kernel.",
};

// The "dark" class is applied unconditionally here (not conditioned
// on prefers-color-scheme) -- this project's own design brief asked
// for a dark-mode-first look matching Vercel/Linear/Cal.com, not a
// theme that flips with the OS. A future round can add a real
// light/dark toggle (persisted via a shadcn-style ThemeProvider) once
// the base dashboard itself is confirmed working end to end.
//
// Build Phase 34: every route is now wrapped in <AppShell>, a
// persistent sidebar (Dashboard / Workflow canvas / Audit log) --
// replacing Build Phase 33's per-page back-link pattern with a real
// application shell.
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background font-sans antialiased">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
