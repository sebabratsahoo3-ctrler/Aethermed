"use client";

import React, { useState } from "react";
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
import { Users, TrendingUp, ShieldAlert, Award, Play, Filter, CheckCircle2 } from "lucide-react";
import { TrialResponse, TrialFilter } from "@/types";

interface ClinicalTrialViewProps {
  trialData: TrialResponse | null;
  onExecuteTrial: (filter: TrialFilter) => void;
  isLoading: boolean;
}

export const ClinicalTrialView: React.FC<ClinicalTrialViewProps> = ({
  trialData,
  onExecuteTrial,
  isLoading,
}) => {
  const [filter, setFilter] = useState<TrialFilter>({
    drug: "Vancomycin",
    standard_dose_mg: 1000.0,
    interval_h: 12,
    min_age: 50.0,
    max_age: 85.0,
    max_egfr: 60.0,
    sample_size: 25,
  });

  const handleRun = () => {
    onExecuteTrial(filter);
  };

  // Build Population Percentile Chart Data
  const chartData = trialData?.population_pk_summary?.time_hours.map((t, idx) => ({
    time: t,
    standard_median: trialData.population_pk_summary.standard_median[idx],
    standard_p10: trialData.population_pk_summary.standard_p10[idx],
    standard_p90: trialData.population_pk_summary.standard_p90[idx],
    cql_median: trialData.population_pk_summary.cql_median[idx],
    cql_p10: trialData.population_pk_summary.cql_p10[idx],
    cql_p90: trialData.population_pk_summary.cql_p90[idx],
  })) || [];

  return (
    <div className="space-y-6">
      {/* Trial Filter & Controls Bar */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <Users className="w-4 h-4 text-purple-400" />
              <span>In-Silico Virtual Clinical Trial Simulator</span>
            </h3>
            <p className="text-xs text-slate-400">
              Randomized in-silico cohort trial evaluating AetherMed CQL vs. Standard Physician Dosing.
            </p>
          </div>
          <button
            type="button"
            onClick={handleRun}
            disabled={isLoading}
            className="flex items-center space-x-2 py-2 px-5 rounded-xl font-semibold text-xs text-white bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 shadow-lg shadow-purple-950/50 disabled:opacity-50 transition-all"
          >
            <Play className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
            <span>{isLoading ? "Simulating Population..." : "Execute Population Trial"}</span>
          </button>
        </div>

        {/* Filter Sliders & Inputs */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
          <div>
            <label className="text-slate-400">Target Drug</label>
            <select
              value={filter.drug}
              onChange={(e) => {
                const d = e.target.value;
                const stdDose = d === "Vancomycin" ? 1000 : d === "Warfarin" ? 5 : 0.125;
                const intH = d === "Vancomycin" ? 12 : 24;
                setFilter({ ...filter, drug: d, standard_dose_mg: stdDose, interval_h: intH });
              }}
              className="w-full glass-input text-xs rounded-lg px-2.5 py-1.5 mt-1 text-slate-200"
            >
              <option value="Vancomycin" className="bg-slate-900">Vancomycin</option>
              <option value="Warfarin" className="bg-slate-900">Warfarin</option>
              <option value="Digoxin" className="bg-slate-900">Digoxin</option>
            </select>
          </div>

          <div>
            <label className="text-slate-400">Standard Dose: <strong className="text-slate-200">{filter.standard_dose_mg} mg</strong></label>
            <input
              type="number"
              step={filter.drug === "Digoxin" ? 0.0625 : 100}
              value={filter.standard_dose_mg}
              onChange={(e) => setFilter({ ...filter, standard_dose_mg: parseFloat(e.target.value) || 100 })}
              className="w-full glass-input text-xs rounded-lg px-2 py-1 mt-1 font-mono text-slate-200"
            />
          </div>

          <div>
            <label className="text-slate-400">Max eGFR: <strong className="text-slate-200">{filter.max_egfr} mL/min</strong></label>
            <input
              type="range"
              min="20"
              max="90"
              value={filter.max_egfr || 60}
              onChange={(e) => setFilter({ ...filter, max_egfr: parseFloat(e.target.value) })}
              className="w-full accent-purple-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer mt-1"
            />
          </div>

          <div>
            <label className="text-slate-400">Min Age: <strong className="text-slate-200">{filter.min_age} yrs</strong></label>
            <input
              type="range"
              min="18"
              max="80"
              value={filter.min_age}
              onChange={(e) => setFilter({ ...filter, min_age: parseFloat(e.target.value) })}
              className="w-full accent-purple-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer mt-1"
            />
          </div>

          <div>
            <label className="text-slate-400">Cohort Sample Size: <strong className="text-slate-200">N={filter.sample_size}</strong></label>
            <input
              type="range"
              min="10"
              max="100"
              step="5"
              value={filter.sample_size}
              onChange={(e) => setFilter({ ...filter, sample_size: parseInt(e.target.value) })}
              className="w-full accent-purple-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer mt-1"
            />
          </div>
        </div>
      </div>

      {trialData ? (
        <>
          {/* Primary & Safety Endpoints Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Primary Endpoint: TTR */}
            <div className="glass-panel p-4 rounded-xl border border-emerald-500/40 bg-emerald-950/20">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider flex items-center space-x-1">
                <Award className="w-3.5 h-3.5 text-emerald-400" />
                <span>Primary Endpoint (TTR)</span>
              </span>
              <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
                {trialData.cql_mean_ttr_pct.toFixed(1)}%{" "}
                <span className="text-xs text-slate-300 font-normal">
                  (vs. {trialData.standard_mean_ttr_pct.toFixed(1)}% Std)
                </span>
              </div>
              <span className="text-xs font-semibold text-emerald-300">
                {trialData.ttr_gain_pct >= 0 ? `+${trialData.ttr_gain_pct.toFixed(1)}%` : `${trialData.ttr_gain_pct.toFixed(1)}%`} TTR Gain
              </span>
            </div>

            {/* Toxicity Incidence Reduction */}
            <div className="glass-panel p-4 rounded-xl border border-rose-500/40 bg-rose-950/20">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider flex items-center space-x-1">
                <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
                <span>Acute Toxicity Incidence</span>
              </span>
              <div className="text-xl font-bold font-mono text-rose-400 mt-1">
                {trialData.cql_nephrotoxicity_incidence_pct.toFixed(1)}%{" "}
                <span className="text-xs text-slate-300 font-normal">
                  (vs. {trialData.standard_nephrotoxicity_incidence_pct.toFixed(1)}% Std)
                </span>
              </div>
              <span className="text-xs font-semibold text-rose-300">
                {trialData.toxicity_relative_reduction_pct.toFixed(1)}% Relative Risk Reduction
              </span>
            </div>

            {/* Sub-cohort Filter Profile */}
            <div className="glass-panel p-4 rounded-xl border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider">Sub-Cohort Profile</span>
              <div className="text-sm font-semibold text-slate-200 mt-1">
                {trialData.filter_description}
              </div>
              <span className="text-xs text-slate-400 font-mono">N = {trialData.sample_size} Virtual Twins</span>
            </div>

            {/* Regimen Protocols */}
            <div className="glass-panel p-4 rounded-xl border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider">Protocol Comparison</span>
              <div className="text-xs text-slate-300 mt-1">
                <strong>Std:</strong> {trialData.standard_physician_regimen}
              </div>
              <div className="text-xs text-purple-300 mt-0.5">
                <strong>CQL:</strong> {trialData.aethermed_cql_regimen}
              </div>
            </div>
          </div>

          {/* Population PK Percentile Envelopes Chart */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-purple-400" />
                  <span>Population PK Trajectory Envelopes (Median &amp; 10th-90th Percentiles)</span>
                </h3>
                <p className="text-xs text-slate-400">
                  Standard Physician Dosing (Slate) vs. AetherMed CQL-Guided Dosing (Purple).
                </p>
              </div>
              <span className="text-xs font-mono text-purple-300">
                Units: {trialData.population_pk_summary.display_unit}
              </span>
            </div>

            <div className="w-full h-72 pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 11, fill: "#94a3b8" }} unit="h" />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(15, 23, 42, 0.95)",
                      borderColor: "#334155",
                      borderRadius: "0.75rem",
                      fontSize: "12px",
                      color: "#f8fafc",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />

                  {/* Shaded Target Window */}
                  <ReferenceArea
                    y1={trialData.population_pk_summary.therapeutic_min}
                    y2={trialData.population_pk_summary.therapeutic_max}
                    fill="#10b981"
                    fillOpacity={0.08}
                    stroke="#10b981"
                    strokeDasharray="3 3"
                    strokeOpacity={0.4}
                  />

                  {/* Standard Regimen Lines */}
                  <Line
                    type="monotone"
                    dataKey="standard_median"
                    name="Standard Median"
                    stroke="#94a3b8"
                    strokeWidth={2}
                    dot={false}
                  />

                  {/* CQL Regimen Lines */}
                  <Line
                    type="monotone"
                    dataKey="cql_median"
                    name="CQL Guided Median"
                    stroke="#a855f7"
                    strokeWidth={2.5}
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Sample Patient Outcomes Breakdown Table */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <Users className="w-4 h-4 text-cyan-400" />
              <span>Virtual Patient Trial Outcomes (Sample N=15)</span>
            </h3>

            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-2.5">Twin ID</th>
                    <th className="p-2.5">Age</th>
                    <th className="p-2.5">eGFR</th>
                    <th className="p-2.5">Std Dose</th>
                    <th className="p-2.5">Std TTR %</th>
                    <th className="p-2.5">Std Trough</th>
                    <th className="p-2.5 text-purple-300">CQL Dose</th>
                    <th className="p-2.5 text-purple-300">CQL TTR %</th>
                    <th className="p-2.5 text-purple-300">CQL Trough</th>
                    <th className="p-2.5">Outcome</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {trialData.patient_outcomes_sample.map((row) => (
                    <tr key={row.patient_id} className="hover:bg-slate-800/30">
                      <td className="p-2.5 font-bold text-slate-200">{row.patient_id}</td>
                      <td className="p-2.5 text-slate-300">{row.age}y</td>
                      <td className="p-2.5 text-cyan-300">{row.egfr.toFixed(1)}</td>
                      <td className="p-2.5 text-slate-400">{row.standard_dose_mg}mg</td>
                      <td className="p-2.5 text-slate-300">{row.standard_ttr_pct.toFixed(0)}%</td>
                      <td className="p-2.5 text-slate-300">{row.standard_trough.toFixed(1)}</td>
                      <td className="p-2.5 font-bold text-purple-300">{row.cql_dose_mg}mg</td>
                      <td className="p-2.5 font-bold text-emerald-400">{row.cql_ttr_pct.toFixed(0)}%</td>
                      <td className="p-2.5 text-purple-200">{row.cql_trough.toFixed(1)}</td>
                      <td className="p-2.5 font-sans">
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950/70 border border-emerald-800/50 text-emerald-300 flex items-center space-x-1 w-fit">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>{row.clinical_outcome}</span>
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        <div className="glass-panel p-12 rounded-2xl border border-slate-800 text-center text-slate-400 space-y-2">
          <Users className="w-8 h-8 text-purple-400 mx-auto animate-pulse" />
          <h4 className="text-sm font-semibold text-slate-200">No Population Trial Executed Yet</h4>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Configure cohort inclusion criteria and click &quot;Execute Population Trial&quot; to benchmark CQL-guided dosing against standard regimens.
          </p>
        </div>
      )}
    </div>
  );
};
