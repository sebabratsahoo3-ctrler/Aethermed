"use client";

import React from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceArea,
} from "recharts";
import { Activity, ShieldCheck, Zap, Droplets, Clock, HeartPulse } from "lucide-react";
import { PKResponse } from "@/types";

interface PKPDViewProps {
  pk: PKResponse | null;
}

export const PKPDView: React.FC<PKPDViewProps> = ({ pk }) => {
  if (!pk) {
    return (
      <div className="glass-panel rounded-2xl p-8 text-center text-slate-400">
        Run simulation to view pharmacokinetic trajectories.
      </div>
    );
  }

  // Sample or downsample points for smooth rendering
  const chartData = pk.time_hours.map((t, idx) => ({
    time: t,
    c_central_neural: pk.c_central_neural[idx] ?? 0,
    c_central_physics: pk.c_central_physics[idx] ?? 0,
    c_peripheral: pk.c_peripheral[idx] ?? 0,
    c_cql: pk.c_cql_recommended ? pk.c_cql_recommended[idx] : undefined,
    therapeutic_min: pk.therapeutic_min,
    therapeutic_max: pk.therapeutic_max,
  }));

  const isNeuralODE = pk.active_model_source.includes("Neural ODE");

  return (
    <div className="space-y-6">
      {/* Top Banner: Active Model Source & Verification */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-cyan-950/80 border border-cyan-700/60 text-cyan-400">
            <Zap className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Active Solver</span>
              <span className="text-[11px] px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">
                Exact Superposition Grid
              </span>
            </div>
            <h3 className="text-sm font-bold text-slate-100">{pk.active_model_source}</h3>
          </div>
        </div>

        <div className="flex items-center space-x-4 text-xs">
          <div className="text-right">
            <span className="text-slate-400">Mass Conservation Error</span>
            <div className="font-mono text-emerald-400 font-semibold">
              &Delta;M = {pk.mass_conservation_error.toExponential(3)}
            </div>
          </div>
          <div className="text-right">
            <span className="text-slate-400">Therapeutic Target</span>
            <div className="font-mono text-cyan-300 font-semibold">
              {pk.therapeutic_min} - {pk.therapeutic_max} {pk.display_unit}
            </div>
          </div>
        </div>
      </div>

      {/* Main PK Trajectory Chart */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Multi-Dose Plasma Concentration Trajectory</span>
            </h3>
            <p className="text-xs text-slate-400">
              Central vs. Peripheral compartmental dynamics with CQL recommended regimen overlay.
            </p>
          </div>
          <div className="text-xs text-slate-400">
            Units: <span className="text-cyan-300 font-mono font-semibold">{pk.display_unit}</span>
          </div>
        </div>

        <div className="w-full h-80 pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="therapeuticArea" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.12} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0.03} />
                </linearGradient>
              </defs>

              <XAxis
                dataKey="time"
                stroke="#64748b"
                tick={{ fontSize: 11, fill: "#94a3b8" }}
                unit="h"
              />
              <YAxis
                stroke="#64748b"
                tick={{ fontSize: 11, fill: "#94a3b8" }}
                unit={` ${pk.display_unit.split("/")[0]}`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgba(15, 23, 42, 0.95)",
                  borderColor: "#334155",
                  borderRadius: "0.75rem",
                  fontSize: "12px",
                  color: "#f8fafc",
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }}
              />

              {/* Shaded Target Therapeutic Range */}
              <ReferenceArea
                y1={pk.therapeutic_min}
                y2={pk.therapeutic_max}
                fill="url(#therapeuticArea)"
                stroke="#10b981"
                strokeDasharray="3 3"
                strokeOpacity={0.4}
              />

              {/* Central Physics ODE Curve */}
              <Line
                type="monotone"
                dataKey="c_central_physics"
                name="Base Physics ODE"
                stroke="#94a3b8"
                strokeDasharray="4 4"
                strokeWidth={1.5}
                dot={false}
              />

              {/* Peripheral Compartment */}
              <Line
                type="monotone"
                dataKey="c_peripheral"
                name="Peripheral Compartment"
                stroke="#f59e0b"
                strokeWidth={1.5}
                dot={false}
              />

              {/* CQL Recommended Overlay (if available) */}
              {pk.c_cql_recommended && (
                <Line
                  type="monotone"
                  dataKey="c_cql"
                  name="CQL Recommended Dose"
                  stroke="#a855f7"
                  strokeWidth={2}
                  strokeDasharray="2 2"
                  dot={false}
                />
              )}

              {/* Central Neural ODE Main Curve */}
              <Line
                type="monotone"
                dataKey="c_central_neural"
                name="Central (Neural PK ODE)"
                stroke="#06b6d4"
                strokeWidth={2.5}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Pharmacokinetic Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        {/* Cmax */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center space-x-1.5 text-xs text-slate-400">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            <span>Peak C<sub>max</sub></span>
          </div>
          <div className="text-lg font-bold text-slate-100 font-mono mt-1">
            {pk.cmax.toFixed(2)}{" "}
            <span className="text-[10px] text-slate-400 font-normal">{pk.display_unit}</span>
          </div>
        </div>

        {/* Cmin / Trough */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center space-x-1.5 text-xs text-slate-400">
            <Droplets className="w-3.5 h-3.5 text-teal-400" />
            <span>Trough C<sub>min</sub></span>
          </div>
          <div className="text-lg font-bold text-slate-100 font-mono mt-1">
            {pk.cmin_trough.toFixed(2)}{" "}
            <span className="text-[10px] text-slate-400 font-normal">{pk.display_unit}</span>
          </div>
        </div>

        {/* AUC24 */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center space-x-1.5 text-xs text-slate-400">
            <Activity className="w-3.5 h-3.5 text-indigo-400" />
            <span>24h Exposure AUC</span>
          </div>
          <div className="text-lg font-bold text-slate-100 font-mono mt-1">
            {pk.auc_24h.toFixed(1)}{" "}
            <span className="text-[10px] text-slate-400 font-normal">
              {pk.display_unit.split("/")[0]}*h/{pk.display_unit.split("/")[1] || "mL"}
            </span>
          </div>
        </div>

        {/* Time-in-Therapeutic-Range (TTR %) */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center space-x-1.5 text-xs text-slate-400">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Target TTR %</span>
          </div>
          <div
            className={`text-lg font-bold font-mono mt-1 ${
              pk.time_in_therapeutic_range_pct >= 60.0 ? "text-emerald-400" : "text-amber-400"
            }`}
          >
            {pk.time_in_therapeutic_range_pct.toFixed(1)}%
          </div>
        </div>

        {/* Half-Life */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center space-x-1.5 text-xs text-slate-400">
            <Clock className="w-3.5 h-3.5 text-purple-400" />
            <span>Elim. Half-Life</span>
          </div>
          <div className="text-lg font-bold text-slate-100 font-mono mt-1">
            {pk.half_life_hours.toFixed(1)}{" "}
            <span className="text-[10px] text-slate-400 font-normal">hrs</span>
          </div>
        </div>

        {/* Clearance */}
        <div className="glass-panel p-3.5 rounded-xl border border-slate-800">
          <div className="flex items-center space-x-1.5 text-xs text-slate-400">
            <HeartPulse className="w-3.5 h-3.5 text-rose-400" />
            <span>Clearance (CL)</span>
          </div>
          <div className="text-lg font-bold text-slate-100 font-mono mt-1">
            {pk.clearance_l_h.toFixed(2)}{" "}
            <span className="text-[10px] text-slate-400 font-normal">L/h</span>
          </div>
        </div>
      </div>
    </div>
  );
};
