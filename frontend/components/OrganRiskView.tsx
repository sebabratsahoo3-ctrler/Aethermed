"use client";

import React, { useState } from "react";
import { ShieldAlert, Activity, GitBranch, Info, AlertOctagon, Heart, Droplet, Stethoscope } from "lucide-react";
import { OrganRiskItem, GraphPayload, GraphNode, GraphEdge } from "@/types";

interface OrganRiskViewProps {
  organRisks: OrganRiskItem[];
  overallRisk: number;
  graphTopology: GraphPayload | null;
}

export const OrganRiskView: React.FC<OrganRiskViewProps> = ({
  organRisks,
  overallRisk,
  graphTopology,
}) => {
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<GraphEdge | null>(null);

  // Positions for 6-node heterogeneous graph layout in SVG (viewBox 0 0 600 380)
  const nodePositions: Record<string, { x: number; y: number; color: string; icon: any }> = {
    Patient: { x: 90, y: 190, color: "#38bdf8", icon: Stethoscope },
    Drug: { x: 230, y: 190, color: "#818cf8", icon: Activity },
    Liver: { x: 370, y: 80, color: "#f59e0b", icon: Droplet },
    Kidney: { x: 370, y: 190, color: "#06b6d4", icon: Activity },
    Cardiovascular: { x: 370, y: 300, color: "#f43f5e", icon: Heart },
    Safety_Monitor: { x: 520, y: 190, color: "#10b981", icon: AlertOctagon },
  };

  const getRiskColor = (score: number) => {
    if (score < 0.33) return "text-emerald-400 bg-emerald-950/70 border-emerald-500/50";
    if (score < 0.67) return "text-amber-400 bg-amber-950/70 border-amber-500/50";
    return "text-rose-400 bg-rose-950/70 border-rose-500/50";
  };

  return (
    <div className="space-y-6">
      {/* 3 Calibrated Organ Risk Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {organRisks.map((item) => {
          const scorePct = Math.round(item.risk_score * 100);
          const isHigh = item.risk_band === "high";
          const isMod = item.risk_band === "moderate";

          return (
            <div
              key={item.organ}
              className={`glass-panel p-5 rounded-2xl border transition-all ${
                isHigh
                  ? "border-rose-500/50 bg-rose-950/20 shadow-lg shadow-rose-950/30"
                  : isMod
                  ? "border-amber-500/40 bg-amber-950/15"
                  : "border-slate-800 bg-slate-900/40"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-slate-100 flex items-center space-x-2">
                  <ShieldAlert
                    className={`w-4 h-4 ${
                      isHigh ? "text-rose-400" : isMod ? "text-amber-400" : "text-emerald-400"
                    }`}
                  />
                  <span>{item.organ.toUpperCase()} SYSTEM</span>
                </span>
                <span
                  className={`text-[11px] font-bold uppercase px-2.5 py-0.5 rounded-full border ${getRiskColor(
                    item.risk_score
                  )}`}
                >
                  {item.risk_band} RISK
                </span>
              </div>

              {/* Score Progress Bar */}
              <div className="mt-4 space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Toxicity Stress Probability</span>
                  <span className="font-mono font-bold text-slate-200">
                    {item.risk_score.toFixed(3)} ({scorePct}%)
                  </span>
                </div>
                <div className="w-full h-2.5 rounded-full bg-slate-950 border border-slate-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      isHigh
                        ? "bg-gradient-to-r from-amber-500 to-rose-500"
                        : isMod
                        ? "bg-gradient-to-r from-emerald-500 to-amber-500"
                        : "bg-emerald-500"
                    }`}
                    style={{ width: `${scorePct}%` }}
                  />
                </div>
              </div>

              {/* Adaptive Conformal Interval */}
              <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase tracking-wider">
                    90% Conformal Interval
                  </span>
                  <span className="font-mono font-semibold text-cyan-300">
                    [{item.interval_lower.toFixed(2)} — {item.interval_upper.toFixed(2)}]
                  </span>
                </div>
                <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950/70 border border-cyan-800/50 text-cyan-400 font-mono">
                  Adaptive CQR
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Interactive Heterogeneous Graph Topology Viewer */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <GitBranch className="w-4 h-4 text-cyan-400" />
              <span>Heterogeneous Physiological Graph Topology (6 Nodes / 16 Edges)</span>
            </h3>
            <p className="text-xs text-slate-400">
              Message-passing pathways with edge attention weights and organ toxicity propagation.
            </p>
          </div>
          {hoveredEdge && (
            <div className="text-xs px-3 py-1 rounded-lg bg-cyan-950/80 border border-cyan-600/50 text-cyan-200 font-mono">
              <strong>{hoveredEdge.source}</strong> &rarr; [{hoveredEdge.relation}] &rarr; <strong>{hoveredEdge.target}</strong> (Attention: {hoveredEdge.attention_weight.toFixed(2)})
            </div>
          )}
        </div>

        {/* SVG Network Graph */}
        <div className="relative w-full h-[380px] bg-slate-950/70 rounded-xl border border-slate-800/60 overflow-hidden flex items-center justify-center">
          <svg className="w-full h-full" viewBox="0 0 600 380">
            {/* Draw Edges */}
            {graphTopology?.edges.map((edge, i) => {
              const srcPos = nodePositions[edge.source];
              const dstPos = nodePositions[edge.target];
              if (!srcPos || !dstPos) return null;

              const isAttentionHigh = edge.attention_weight > 0.7;
              const isHovered = hoveredEdge === edge;

              // Arc curved lines to distinguish bidirectional edges
              const dx = dstPos.x - srcPos.x;
              const dy = dstPos.y - srcPos.y;
              const cx = (srcPos.x + dstPos.x) / 2 + (i % 2 === 0 ? -12 : 12);
              const cy = (srcPos.y + dstPos.y) / 2 + (i % 2 === 0 ? 12 : -12);

              return (
                <g key={i}>
                  <path
                    d={`M ${srcPos.x} ${srcPos.y} Q ${cx} ${cy} ${dstPos.x} ${dstPos.y}`}
                    fill="none"
                    stroke={
                      isHovered
                        ? "#06b6d4"
                        : isAttentionHigh
                        ? "#38bdf8"
                        : "#334155"
                    }
                    strokeWidth={isHovered ? 3 : Math.max(1, edge.attention_weight * 3.5)}
                    strokeOpacity={isHovered ? 1.0 : isAttentionHigh ? 0.75 : 0.35}
                    className="cursor-pointer transition-all hover:stroke-cyan-400"
                    onMouseEnter={() => setHoveredEdge(edge)}
                    onMouseLeave={() => setHoveredEdge(null)}
                  />
                </g>
              );
            })}

            {/* Draw Nodes */}
            {graphTopology?.nodes.map((node) => {
              const pos = nodePositions[node.id];
              if (!pos) return null;
              const isSelected = selectedNode?.id === node.id;

              return (
                <g
                  key={node.id}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  className="cursor-pointer group"
                  onClick={() => setSelectedNode(node)}
                >
                  {/* Outer Pulsing Aura for Monitor or High Risk */}
                  {node.risk_score && node.risk_score > 0.6 && (
                    <circle
                      r="28"
                      fill="none"
                      stroke="#f43f5e"
                      strokeWidth="1.5"
                      className="animate-ping opacity-30"
                    />
                  )}

                  {/* Node Circle */}
                  <circle
                    r="22"
                    fill="#0f172a"
                    stroke={isSelected ? "#06b6d4" : pos.color}
                    strokeWidth={isSelected ? 3 : 2}
                    className="transition-all group-hover:scale-110 shadow-lg"
                  />

                  {/* Node Icon / Text */}
                  <text
                    y="4"
                    textAnchor="middle"
                    fill="#f8fafc"
                    fontSize="10"
                    fontWeight="bold"
                    className="select-none pointer-events-none"
                  >
                    {node.id.slice(0, 3).toUpperCase()}
                  </text>

                  {/* Node Label Below */}
                  <text
                    y="36"
                    textAnchor="middle"
                    fill="#94a3b8"
                    fontSize="10"
                    className="select-none font-medium"
                  >
                    {node.label}
                  </text>

                  {/* Risk Badge if available */}
                  {node.risk_score !== undefined && node.risk_score !== null && (
                    <text
                      y="48"
                      textAnchor="middle"
                      fill={node.risk_score > 0.6 ? "#f43f5e" : "#38bdf8"}
                      fontSize="9"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      {(node.risk_score * 100).toFixed(0)}%
                    </text>
                  )}
                </g>
              );
            })}
          </svg>

          {/* Node Detail Popup */}
          {selectedNode && (
            <div className="absolute bottom-3 left-3 right-3 p-3 rounded-xl glass-panel border border-cyan-600/50 shadow-xl flex items-center justify-between text-xs">
              <div>
                <strong className="text-cyan-300">{selectedNode.label}</strong> ({selectedNode.type})
                <div className="text-slate-400 mt-0.5">
                  Raw Biomarkers: [{selectedNode.features.map((f) => f.toFixed(1)).join(", ")}]
                </div>
              </div>
              <button
                type="button"
                onClick={() => setSelectedNode(null)}
                className="text-slate-400 hover:text-slate-200 text-xs px-2 py-1 rounded bg-slate-800"
              >
                Close
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
