"use client";

import React from "react";
import { Compass, TrendingUp, CheckCircle, AlertCircle, BarChart3, Database } from "lucide-react";
import { RLResponse } from "@/types";

interface RLPolicyViewProps {
  rl: RLResponse | null;
  drug: string;
}

export const RLPolicyView: React.FC<RLPolicyViewProps> = ({ rl, drug }) => {
  if (!rl) {
    return (
      <div className="glass-panel rounded-2xl p-8 text-center text-slate-400">
        Run simulation to view offline reinforcement learning dosing policy recommendations.
      </div>
    );
  }

  const isDoseReduced = rl.dose_delta_mg < 0;
  const isDoseIncreased = rl.dose_delta_mg > 0;

  return (
    <div className="space-y-6">
      {/* Side-by-side Dosing Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Baseline Physician Dose */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <Compass className="w-4 h-4 text-slate-400" />
            <span className="uppercase tracking-wider font-semibold">Physician Baseline</span>
          </div>
          <div className="text-2xl font-bold font-mono text-slate-200 mt-2">
            {rl.baseline_dose_mg.toFixed(1)} <span className="text-sm font-normal text-slate-400">mg</span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Starting empirical clinical dose</p>
        </div>

        {/* CQL Optimal Recommended Dose */}
        <div className="glass-panel p-5 rounded-2xl border border-cyan-500/50 bg-cyan-950/20 shadow-lg shadow-cyan-950/40">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-xs text-cyan-300">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              <span className="uppercase tracking-wider font-semibold">CQL Optimal Policy</span>
            </div>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950 border border-cyan-700 text-cyan-300 font-mono">
              Offline RL
            </span>
          </div>
          <div className="text-2xl font-bold font-mono text-cyan-300 mt-2 flex items-baseline space-x-2">
            <span>{rl.cql_recommended_dose_mg.toFixed(1)} mg</span>
            <span
              className={`text-xs font-semibold px-2 py-0.5 rounded ${
                isDoseReduced
                  ? "bg-rose-950/80 text-rose-300 border border-rose-800"
                  : isDoseIncreased
                  ? "bg-emerald-950/80 text-emerald-300 border border-emerald-800"
                  : "bg-slate-800 text-slate-300"
              }`}
            >
              {rl.dose_delta_mg >= 0 ? `+${rl.dose_delta_mg.toFixed(1)}` : rl.dose_delta_mg.toFixed(1)} mg
            </span>
          </div>
          <p className="text-xs text-cyan-200/80 mt-1">
            Conservative Q-learning policy with toxicity penalty
          </p>
        </div>

        {/* PK-Target Analytical Dose */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800">
          <div className="flex items-center space-x-2 text-xs text-indigo-300">
            <BarChart3 className="w-4 h-4 text-indigo-400" />
            <span className="uppercase tracking-wider font-semibold">Target AUC Analytical</span>
          </div>
          <div className="text-2xl font-bold font-mono text-indigo-200 mt-2">
            {rl.analytical_target_dose_mg.toFixed(1)} <span className="text-sm font-normal text-slate-400">mg</span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Direct steady-state target exposure titrator</p>
        </div>
      </div>

      {/* Clinical Rationale Box */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex items-start space-x-3.5">
        <CheckCircle className="w-5 h-5 text-cyan-400 mt-0.5 flex-shrink-0" />
        <div className="space-y-1">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-cyan-300">
            Clinical Safety & Titration Rationale
          </h4>
          <p className="text-xs text-slate-300 leading-relaxed">{rl.recommendation_rationale}</p>
        </div>
      </div>

      {/* Off-Policy Evaluation (OPE) Benchmarking Cards */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <Database className="w-4 h-4 text-indigo-400" />
              <span>Offline RL Policy Evaluation (OPE Suite)</span>
            </h3>
            <p className="text-xs text-slate-400">
              Rigorous counterfactual validation against logged historical dosing trajectories.
            </p>
          </div>
          <span className="text-xs font-mono text-indigo-300 bg-indigo-950/80 border border-indigo-800/60 px-2.5 py-1 rounded-lg">
            ESS: {rl.ope_metrics.effective_sample_size.toFixed(0)} Transitions
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {/* Doubly Robust Return */}
          <div className="glass-panel p-4 rounded-xl border border-emerald-500/40 bg-emerald-950/15">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider">Doubly Robust (DR)</span>
            <div className="text-lg font-bold font-mono text-emerald-400 mt-1">
              {rl.ope_metrics.doubly_robust_return.toFixed(1)}
            </div>
            <span className="text-[10px] text-emerald-300 font-semibold">
              +{rl.ope_metrics.pct_improvement_dr.toFixed(1)}% vs. Baseline
            </span>
          </div>

          {/* Weighted Importance Sampling (WIS) */}
          <div className="glass-panel p-4 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider">WIS Return</span>
            <div className="text-lg font-bold font-mono text-slate-200 mt-1">
              {rl.ope_metrics.weighted_importance_sampling_return.toFixed(1)}
            </div>
            <span className="text-[10px] text-slate-400">Normalized Trajectory Weights</span>
          </div>

          {/* Fitted Q-Evaluation (FQE) */}
          <div className="glass-panel p-4 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider">FQE Return</span>
            <div className="text-lg font-bold font-mono text-slate-200 mt-1">
              {rl.ope_metrics.fitted_q_evaluation_return.toFixed(1)}
            </div>
            <span className="text-[10px] text-slate-400">Neural Critic Rollout</span>
          </div>

          {/* Behavior Baseline Return */}
          <div className="glass-panel p-4 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider">Behavior Baseline</span>
            <div className="text-lg font-bold font-mono text-slate-400 mt-1">
              {rl.ope_metrics.behavior_baseline_return.toFixed(1)}
            </div>
            <span className="text-[10px] text-slate-500">Historical Physician Return</span>
          </div>
        </div>
      </div>
    </div>
  );
};
