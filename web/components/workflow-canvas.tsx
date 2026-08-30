"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import {
  BaseEdge,
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  getBezierPath,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Brain } from "lucide-react";

import type { AgentSummary } from "@/lib/api";
import { iconForAgent } from "@/lib/icons";

// Build Phase 33 (Part 2), redesigned in Build Phase 34, then again in
// Build Phase 36: the visual node-based agent-workflow view -- the UI
// counterpart this project's own already-real
// `WorkflowDefinition`/`WorkflowStep` (core/kernel/kernel.py, wired
// via core/kernel/workflow_config.py) never had until now. Built with
// @xyflow/react (MIT-licensed, github.com/xyflow/xyflow) --
// deliberately NOT any code from Comfy-Org/ComfyUI (GPL-3.0) -- see
// this project's own conversation history for the full licensing
// research behind that choice.
//
// Attribution: the small "React Flow" badge rendered in the corner of
// the canvas is intentionally left visible. @xyflow/react's own docs
// (reactflow.dev/remove-attribution) are explicit that hiding it via
// `proOptions.hideAttribution` is a paid Pro-subscription feature --
// separate from and in addition to the library's underlying MIT code
// license, which only covers the code itself, not this branding
// condition. Per this project's standing "minimize real financial
// cost, self-host for free where possible" rule, we do not pay for
// that -- the badge stays on screen (it does not add a dependency,
// a server call, or any functional limitation).
//
// Build Phase 36: the user asked specifically for a hub-and-spoke
// layout -- the Kernel as a living, glowing core at the center, with
// the registered agents arranged radially around it, connected by a
// beam that a command visibly travels through, rather than the
// left-to-right pipeline row from Build Phase 33/34. The Kernel node
// (KernelNode below) is a new, non-agent node with its own pulsing
// glow animation (framer-motion, looping scale/opacity/box-shadow --
// "حية وليست ثابتة" as the user put it). The connecting beam
// (BeamEdge below) uses native SVG `<animateMotion>` along the edge's
// own bezier path -- the standard, well-documented xyflow technique
// for "a marker traveling along an edge" (reactflow.dev's own
// "Animating Edges" example), not a hand-rolled JS animation loop.
//
// Honest scope, matching every prior Build Phase's own "state what's
// NOT done yet" convention: this is still a STATIC view of the three
// registered agents -- it does not yet read a real
// WorkflowDefinition/WorkflowStep configuration, is not yet editable
// (no drag-to-connect a NEW pipeline), and does not yet show live
// execution status per node (no WebSocket wiring to a running
// Kernel.run() call).
const AGENT_NODE_WIDTH = 220;
const RADIUS = 260;
const KERNEL_SIZE = 96;

interface AgentNodeData extends Record<string, unknown> {
  label: string;
  description: string;
  subject: string;
  targetPosition: Position;
}

function AgentNode({ data }: NodeProps) {
  const { label, description, subject, targetPosition } =
    data as unknown as AgentNodeData;
  const Icon = iconForAgent(subject);

  return (
    <div
      className="group rounded-xl border border-border bg-card/90 p-3 shadow-lg backdrop-blur transition-shadow hover:shadow-primary/20"
      style={{ width: AGENT_NODE_WIDTH }}
    >
      <Handle type="target" position={targetPosition} className="!bg-primary" />
      <div className="flex items-start gap-2.5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-violet-500 shadow-md shadow-primary/30">
          <Icon className="h-4 w-4 text-white" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-foreground">{label}</p>
          <p className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
            {description}
          </p>
        </div>
      </div>
    </div>
  );
}

// The central "core" -- a radiant, continuously-pulsing node rather
// than a static box, matching the user's own description ("نواة
// مشعة ... اضاءتها حية"). Exposes one source handle on each of its
// four sides (kept functionally real but visually hidden -- opacity
// 0 -- since the glow itself already reads as the node's boundary) so
// buildNodesAndEdges can route each spoke from whichever side faces
// that agent.
function KernelNode() {
  return (
    <div
      className="relative flex items-center justify-center"
      style={{ width: KERNEL_SIZE, height: KERNEL_SIZE }}
    >
      <Handle id="top" type="source" position={Position.Top} className="opacity-0" />
      <Handle id="right" type="source" position={Position.Right} className="opacity-0" />
      <Handle id="bottom" type="source" position={Position.Bottom} className="opacity-0" />
      <Handle id="left" type="source" position={Position.Left} className="opacity-0" />

      {[0, 1].map((ring) => (
        <motion.span
          key={ring}
          className="absolute inset-0 rounded-full bg-primary/40"
          animate={{ scale: [1, 1.7, 1], opacity: [0.55, 0, 0.55] }}
          transition={{
            duration: 2.6,
            repeat: Infinity,
            ease: "easeInOut",
            delay: ring * 1.3,
          }}
        />
      ))}

      <motion.div
        className="relative flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-primary via-violet-500 to-fuchsia-500"
        animate={{
          boxShadow: [
            "0 0 20px 4px hsl(217 91% 60% / 0.55)",
            "0 0 42px 12px hsl(270 91% 65% / 0.65)",
            "0 0 20px 4px hsl(217 91% 60% / 0.55)",
          ],
        }}
        transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut" }}
      >
        <Brain className="h-7 w-7 text-white" />
      </motion.div>

      <p className="absolute -bottom-6 whitespace-nowrap text-xs font-medium text-muted-foreground">
        Kernel
      </p>
    </div>
  );
}

const NODE_TYPES = { agentNode: AgentNode, kernelNode: KernelNode };

// A command visibly traveling through the connector, per the user's
// own request ("ذلك الربط في حد ذاته يمشي فيه شعاع كأن الاوامر تمر
// داخله"). `<animateMotion path={edgePath} />` is native SVG, driven
// by the browser's own animation engine -- not React state -- so it
// stays smooth regardless of React re-renders.
function BeamEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  style,
}: EdgeProps) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{ stroke: "hsl(240 5% 34%)", strokeWidth: 1.5, ...style }}
      />
      <circle r={4} fill="hsl(217 91% 65%)" style={{ filter: "drop-shadow(0 0 5px hsl(217 91% 60%))" }}>
        <animateMotion dur="2.2s" repeatCount="indefinite" path={edgePath} />
      </circle>
      <circle r={3} fill="hsl(280 91% 72%)" style={{ filter: "drop-shadow(0 0 5px hsl(280 91% 70%))" }}>
        <animateMotion dur="2.2s" repeatCount="indefinite" path={edgePath} begin="1.1s" />
      </circle>
    </>
  );
}

const EDGE_TYPES = { beamEdge: BeamEdge };

// Given the offset from the kernel's center to an agent's center,
// picks whichever cardinal side (top/right/bottom/left) the agent
// mostly sits on -- used for both the kernel's outgoing handle and
// the agent's incoming handle, so each spoke enters/leaves from a
// sensible side instead of every edge bunching into one corner.
function nearestSide(dx: number, dy: number): {
  kernelHandleId: string;
  agentTargetPosition: Position;
} {
  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0
      ? { kernelHandleId: "right", agentTargetPosition: Position.Left }
      : { kernelHandleId: "left", agentTargetPosition: Position.Right };
  }
  return dy >= 0
    ? { kernelHandleId: "bottom", agentTargetPosition: Position.Top }
    : { kernelHandleId: "top", agentTargetPosition: Position.Bottom };
}

function buildNodesAndEdges(agents: AgentSummary[]): {
  nodes: Node[];
  edges: Edge[];
} {
  const kernelNode: Node = {
    id: "__kernel__",
    type: "kernelNode",
    position: { x: -KERNEL_SIZE / 2, y: -KERNEL_SIZE / 2 },
    draggable: false,
    selectable: false,
    data: {},
  };

  const nodes: Node[] = [kernelNode];
  const edges: Edge[] = [];

  agents.forEach((agent, index) => {
    // Start at the top (-90deg) and go clockwise, evenly spaced.
    const angle = (2 * Math.PI * index) / agents.length - Math.PI / 2;
    const dx = RADIUS * Math.cos(angle);
    const dy = RADIUS * Math.sin(angle);
    const { kernelHandleId, agentTargetPosition } = nearestSide(dx, dy);

    nodes.push({
      id: agent.subject,
      type: "agentNode",
      position: { x: dx - AGENT_NODE_WIDTH / 2, y: dy - 40 },
      data: {
        label: agent.display_name,
        description: agent.description,
        subject: agent.subject,
        targetPosition: agentTargetPosition,
      },
    });

    edges.push({
      id: `__kernel__->${agent.subject}`,
      source: "__kernel__",
      sourceHandle: kernelHandleId,
      target: agent.subject,
      type: "beamEdge",
    });
  });

  return { nodes, edges };
}

export function WorkflowCanvas({ agents }: { agents: AgentSummary[] }) {
  const { nodes, edges } = useMemo(() => buildNodesAndEdges(agents), [agents]);

  return (
    <div className="h-[600px] w-full overflow-hidden rounded-xl border border-border bg-card/20">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        edgeTypes={EDGE_TYPES}
        fitView
      >
        <Background variant={BackgroundVariant.Dots} gap={24} color="hsl(240 4% 30%)" />
        <Controls />
        <MiniMap
          pannable
          zoomable
          maskColor="hsl(240 10% 3.9% / 0.7)"
          nodeColor="hsl(217 91% 60%)"
        />
      </ReactFlow>
    </div>
  );
}
