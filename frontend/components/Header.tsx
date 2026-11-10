"use client";

import React from "react";
import { Activity, ShieldCheck, AlertTriangle, Cpu, Database, Layers } from "lucide-react";
import { HealthResponse, SafetyOODResponse } from "@/types";

interface HeaderProps {
  health: HealthResponse | null;
  safety: SafetyOODResponse | null;
}

export const Header: React.FC<HeaderProps> = ({ health, safety }) => {
  const isHealthy = health?.status === "healthy";
  const isOOD = safety?.is_ood || false;

  return (
    <header className="w-full glass-panel border-b border-slate-800/80 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-50 shadow-xl">
      {/* Brand & Platform */}
      <div className="flex items-center space-x-3.5">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 via-teal-500/20 to-emerald-500/10 border border-cyan-500/40 shadow-lg shadow-cyan-950/50">
          <Activity className="w-6 h-6 text-cyan-400 animate-pulse" />
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
          </span>
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-slate-100 via-cyan-200 to-teal-300 bg-clip-text text-transparent">
              AETHERMED <span className="text-cyan-400 text-xs font-semibold px-2 py-0.5 rounded-full bg-cyan-950/70 border border-cyan-500/40">4.0 PRO</span>
            </h1>
          </div>
          <p className="text-xs text-slate-400">
            Next-Gen Precision TDM & In-Silico Clinical Trial Cockpit
          </p>
        </div>
      </div>

      {/* Telemetry & Safety Status Badges */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Backend & Device Indicator */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300">
          <div className={`w-2 h-2 rounded-full ${isHealthy ? "bg-emerald-400 shadow-sm shadow-emerald-400" : "bg-rose-500"}`} />
          <span className="font-medium text-slate-200">API: {isHealthy ? "Online" : "Connecting..."}</span>
          <span className="text-slate-600">|</span>
          <span className="flex items-center space-x-1 text-cyan-300 font-mono">
            <Cpu className="w-3.5 h-3.5 inline text-cyan-400" />
            <span>{health?.device?.toUpperCase() || "CPU"}</span>
          </span>
        </div>

        {/* ChromaDB Chunks */}
        <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300">
          <Database className="w-3.5 h-3.5 text-indigo-400" />
          <span>Corpus:</span>
          <span className="font-semibold text-indigo-300">{health?.chroma_chunk_count || 10} chunks</span>
        </div>

        {/* Loaded Twins */}
        <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300">
          <Layers className="w-3.5 h-3.5 text-teal-400" />
          <span>Cohort:</span>
          <span className="font-semibold text-teal-300">{health?.cohort_twins_count || 1000} twins</span>
        </div>

        {/* OOD Safety Guard Badge */}
        {safety && (
          <div
            className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-medium border shadow-md transition-all ${
              isOOD
                ? "bg-rose-950/70 border-rose-500/60 text-rose-200 animate-pulse"
                : "bg-emerald-950/60 border-emerald-500/50 text-emerald-200"
            }`}
          >
            {isOOD ? (
              <AlertTriangle className="w-4 h-4 text-rose-400" />
            ) : (
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            )}
            <span>
              {isOOD ? (
                <>
                  <strong className="text-rose-300">OOD ALERT:</strong> D<sub>M</sub>={safety.mahalanobis_distance.toFixed(2)} &gt; &tau;={safety.threshold.toFixed(2)}
                </>
              ) : (
                <>
                  <strong>IN-BOUNDS:</strong> D<sub>M</sub>={safety.mahalanobis_distance.toFixed(2)} &le; &tau;={safety.threshold.toFixed(2)}
                </>
              )}
            </span>
          </div>
        )}
      </div>
    </header>
  );
};
