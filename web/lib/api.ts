// Thin, typed client for the api/ FastAPI backend (Build Phase 33
// Part 1) -- every shape here mirrors api/schemas.py exactly. This
// file contains no business logic, matching that backend's own "thin
// translation layer" convention: parse/serialize only.

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface ComponentStatus {
  name: string;
  configured: boolean;
  detail: string;
}

export interface SystemStatus {
  components: ComponentStatus[];
  all_configured: boolean;
}

export interface AgentSummary {
  subject: string;
  display_name: string;
  description: string;
}

export interface KernelRunResult {
  status: string;
  subject: string | null;
  reason: string | null;
  recovery_attempts: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
}

export interface AuditEvent {
  [key: string]: unknown;
}

class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    cache: "no-store",
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(
      typeof body?.detail === "string" ? body.detail : response.statusText,
      response.status
    );
  }

  return response.json() as Promise<T>;
}

export function getSystemStatus(): Promise<SystemStatus> {
  return request<SystemStatus>("/system/status");
}

export function getAgents(): Promise<AgentSummary[]> {
  return request<AgentSummary[]>("/agents");
}

export function runKernelTask(task: string): Promise<KernelRunResult> {
  return request<KernelRunResult>("/kernel/run", {
    method: "POST",
    body: JSON.stringify({ task }),
  });
}

export function getRecentAuditEvents(limit = 50): Promise<AuditEvent[]> {
  return request<AuditEvent[]>(`/audit-log/recent?limit=${limit}`);
}

export { ApiError, API_BASE_URL };
