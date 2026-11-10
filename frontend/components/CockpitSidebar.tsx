"use client";

import React, { useState } from "react";
import { User, Pill, Activity, Plus, Trash2, RefreshCw, Sparkles, CheckCircle2 } from "lucide-react";
import { PatientTwinInput, TDMPoint, CohortPatient } from "@/types";

interface CockpitSidebarProps {
  input: PatientTwinInput;
  onChange: (updated: PatientTwinInput) => void;
  onSimulate: () => void;
  onRunTrial: () => void;
  isSimulating: boolean;
  sampleCohort?: CohortPatient[];
}

export const CockpitSidebar: React.FC<CockpitSidebarProps> = ({
  input,
  onChange,
  onSimulate,
  onRunTrial,
  isSimulating,
  sampleCohort = [],
}) => {
  const [newTdmTime, setNewTdmTime] = useState<string>("24");
  const [newTdmConc, setNewTdmConc] = useState<string>("");

  const updateField = <K extends keyof PatientTwinInput>(field: K, value: PatientTwinInput[K]) => {
    onChange({ ...input, [field]: value });
  };

  const handleDrugChange = (drug: string) => {
    let defaultDose = 5.0;
    let defaultInterval = 24;
    let loadingDose: number | undefined = undefined;

    if (drug === "Vancomycin") {
      defaultDose = 1000.0;
      defaultInterval = 12;
      loadingDose = 1500.0;
    } else if (drug === "Digoxin") {
      defaultDose = 0.125;
      defaultInterval = 24;
      loadingDose = 0.25;
    }

    onChange({
      ...input,
      drug,
      dose_mg: defaultDose,
      interval_h: defaultInterval,
      loading_dose_mg: loadingDose,
    });
  };

  const handleRecomputeEGFR = () => {
    // 2021 CKD-EPI Creatinine Equation without race
    const scr = input.serum_creatinine;
    const isFemale = input.gender.toLowerCase() === "female";
    const kappa = isFemale ? 0.7 : 0.9;
    const alpha = isFemale ? -0.241 : -0.302;
    const genderFactor = isFemale ? 1.012 : 1.0;

    const scrRatio = scr / kappa;
    const minPart = Math.min(scrRatio, 1.0);
    const maxPart = Math.max(scrRatio, 1.0);

    const calc = 142.0 * Math.pow(minPart, alpha) * Math.pow(maxPart, -1.20) * Math.pow(0.9938, input.age) * genderFactor;
    updateField("egfr", Math.round(Math.min(140.0, Math.max(5.0, calc)) * 10) / 10);
  };

  const handleAddTDM = () => {
    const t = parseFloat(newTdmTime);
    const c = parseFloat(newTdmConc);
    if (!isNaN(t) && !isNaN(c) && c > 0) {
      updateField("tdm_observations", [...input.tdm_observations, { time_h: t, concentration: c }]);
      setNewTdmConc("");
    }
  };

  const handleRemoveTDM = (idx: number) => {
    const updated = input.tdm_observations.filter((_, i) => i !== idx);
    updateField("tdm_observations", updated);
  };

  const handleSelectCohortPatient = (patient: CohortPatient) => {
    onChange({
      ...input,
      patient_id: patient.patient_id,
      age: patient.age,
      gender: patient.gender,
      weight: patient.weight,
      serum_creatinine: patient.serum_creatinine,
      egfr: patient.egfr,
      alt: patient.alt,
      ast: patient.ast,
      cyp2c9_status: patient.cyp2c9_status || "*1/*1",
      vkorc1_status: patient.vkorc1_status || "G/G",
      potassium: patient.serum_potassium || 4.2,
      albumin: patient.albumin || 4.1,
    });
  };

  return (
    <aside className="w-full lg:w-96 flex flex-col gap-4 p-4 glass-panel rounded-2xl border border-slate-800 shadow-2xl h-fit">
      {/* Header & Cohort Roster */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
        <div className="flex items-center space-x-2">
          <User className="w-5 h-5 text-cyan-400" />
          <h2 className="font-semibold text-slate-100 text-sm tracking-wide">DIGITAL TWIN COCKPIT</h2>
        </div>
        <span className="text-xs font-mono text-cyan-400 px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-800/50">
          {input.patient_id}
        </span>
      </div>

      {/* Cohort Preset Quick Selector */}
      {sampleCohort.length > 0 && (
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-slate-400 flex items-center justify-between">
            <span>Load Twin from Synthetic Cohort</span>
            <span className="text-[10px] text-cyan-400">1,000 Verified</span>
          </label>
          <select
            className="w-full text-xs rounded-lg glass-input px-3 py-2 text-slate-200 cursor-pointer"
            onChange={(e) => {
              const selected = sampleCohort.find((p) => p.patient_id === e.target.value);
              if (selected) handleSelectCohortPatient(selected);
            }}
            value={input.patient_id}
          >
            {sampleCohort.slice(0, 20).map((p) => (
              <option key={p.patient_id} value={p.patient_id} className="bg-slate-900 text-slate-200">
                {p.patient_id} — {p.gender === "female" ? "F" : "M"}, {p.age}y, eGFR {p.egfr.toFixed(0)}, Cr {p.serum_creatinine.toFixed(2)}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Demographics & Biometrics */}
      <div className="space-y-3 pt-1">
        <div className="text-xs font-semibold text-cyan-300 uppercase tracking-wider flex items-center space-x-1.5">
          <span>Patient Demographics & Labs</span>
        </div>

        <div className="grid grid-cols-2 gap-2.5 text-xs">
          <div>
            <label className="text-slate-400">Age: <strong className="text-slate-200">{input.age} yrs</strong></label>
            <input
              type="range"
              min="18"
              max="95"
              value={input.age}
              onChange={(e) => updateField("age", parseFloat(e.target.value))}
              className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>

          <div>
            <label className="text-slate-400">Weight: <strong className="text-slate-200">{input.weight} kg</strong></label>
            <input
              type="range"
              min="40"
              max="140"
              step="0.5"
              value={input.weight}
              onChange={(e) => updateField("weight", parseFloat(e.target.value))}
              className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2.5 text-xs">
          <div>
            <label className="text-slate-400">Gender</label>
            <select
              value={input.gender}
              onChange={(e) => updateField("gender", e.target.value)}
              className="w-full glass-input text-xs rounded-lg px-2.5 py-1.5 mt-1 text-slate-200"
            >
              <option value="female" className="bg-slate-900">Female</option>
              <option value="male" className="bg-slate-900">Male</option>
            </select>
          </div>

          <div>
            <label className="text-slate-400">Creatinine: <strong className="text-slate-200">{input.serum_creatinine} mg/dL</strong></label>
            <input
              type="number"
              step="0.05"
              min="0.3"
              max="6.0"
              value={input.serum_creatinine}
              onChange={(e) => updateField("serum_creatinine", parseFloat(e.target.value) || 1.0)}
              className="w-full glass-input text-xs rounded-lg px-2.5 py-1 mt-1 text-slate-200"
            />
          </div>
        </div>

        {/* eGFR with 2021 CKD-EPI Recompute */}
        <div className="p-2.5 rounded-xl bg-slate-900/70 border border-slate-800 flex items-center justify-between gap-2">
          <div>
            <div className="text-[11px] text-slate-400">Renal Function (eGFR)</div>
            <div className="text-sm font-bold text-cyan-400 font-mono">
              {(input.egfr || 65.0).toFixed(1)} <span className="text-[10px] text-slate-400 font-normal">mL/min</span>
            </div>
          </div>
          <button
            type="button"
            onClick={handleRecomputeEGFR}
            className="flex items-center space-x-1 text-[11px] px-2.5 py-1 rounded-lg bg-cyan-950/80 hover:bg-cyan-900 border border-cyan-700/50 text-cyan-300 transition-colors"
            title="Recalculate with 2021 CKD-EPI Creatinine Equation"
          >
            <RefreshCw className="w-3 h-3 animate-spin-reverse" />
            <span>2021 CKD-EPI</span>
          </button>
        </div>

        {/* Liver Panels */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <label className="text-slate-400">ALT: <strong className="text-slate-200">{input.alt} U/L</strong></label>
            <input
              type="range"
              min="10"
              max="200"
              value={input.alt}
              onChange={(e) => updateField("alt", parseFloat(e.target.value))}
              className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>
          <div>
            <label className="text-slate-400">AST: <strong className="text-slate-200">{input.ast} U/L</strong></label>
            <input
              type="range"
              min="10"
              max="200"
              value={input.ast}
              onChange={(e) => updateField("ast", parseFloat(e.target.value))}
              className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
            />
          </div>
        </div>

        {/* Pharmacogenomics */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <label className="text-slate-400">CYP2C9</label>
            <select
              value={input.cyp2c9_status}
              onChange={(e) => updateField("cyp2c9_status", e.target.value)}
              className="w-full glass-input text-xs rounded-lg px-2 py-1 mt-1 text-slate-200"
            >
              <option value="*1/*1" className="bg-slate-900">*1/*1 (Normal)</option>
              <option value="*1/*2" className="bg-slate-900">*1/*2 (Intermediate)</option>
              <option value="*1/*3" className="bg-slate-900">*1/*3 (Intermediate)</option>
              <option value="*2/*2" className="bg-slate-900">*2/*2 (Poor)</option>
              <option value="*3/*3" className="bg-slate-900">*3/*3 (Poor/Toxic)</option>
            </select>
          </div>
          <div>
            <label className="text-slate-400">VKORC1</label>
            <select
              value={input.vkorc1_status}
              onChange={(e) => updateField("vkorc1_status", e.target.value)}
              className="w-full glass-input text-xs rounded-lg px-2 py-1 mt-1 text-slate-200"
            >
              <option value="G/G" className="bg-slate-900">G/G (Normal)</option>
              <option value="G/A" className="bg-slate-900">G/A (Sensitive)</option>
              <option value="A/A" className="bg-slate-900">A/A (Hyper-Sensitive)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Target Drug & Regimen Configurator */}
      <div className="space-y-3 pt-2 border-t border-slate-800/80">
        <div className="text-xs font-semibold text-cyan-300 uppercase tracking-wider flex items-center space-x-1.5">
          <Pill className="w-4 h-4 text-cyan-400" />
          <span>Drug & Dosing Regimen</span>
        </div>

        <div>
          <label className="text-xs text-slate-400">Target NTI Drug</label>
          <div className="grid grid-cols-3 gap-1.5 mt-1">
            {["Warfarin", "Vancomycin", "Digoxin"].map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => handleDrugChange(d)}
                className={`py-1.5 px-2 text-xs rounded-lg font-medium transition-all ${
                  input.drug === d
                    ? "bg-cyan-500/20 border border-cyan-400 text-cyan-200 shadow-sm shadow-cyan-900/50"
                    : "bg-slate-900/70 border border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>

        {/* Dose & Interval */}
        <div className="grid grid-cols-2 gap-2.5 text-xs">
          <div>
            <label className="text-slate-400">
              Dose: <strong className="text-slate-200">{input.dose_mg} mg</strong>
            </label>
            <input
              type="number"
              step={input.drug === "Digoxin" ? 0.0625 : input.drug === "Warfarin" ? 0.5 : 125}
              value={input.dose_mg}
              onChange={(e) => updateField("dose_mg", parseFloat(e.target.value) || 1.0)}
              className="w-full glass-input text-xs rounded-lg px-2.5 py-1 mt-1 text-slate-200 font-mono"
            />
          </div>

          <div>
            <label className="text-slate-400">Interval (&tau;)</label>
            <select
              value={input.interval_h}
              onChange={(e) => updateField("interval_h", parseInt(e.target.value))}
              className="w-full glass-input text-xs rounded-lg px-2 py-1 mt-1 text-slate-200"
            >
              <option value="8" className="bg-slate-900">q8h (TID)</option>
              <option value="12" className="bg-slate-900">q12h (BID)</option>
              <option value="24" className="bg-slate-900">q24h (Daily)</option>
            </select>
          </div>
        </div>

        {/* Duration */}
        <div className="text-xs">
          <label className="text-slate-400">
            Simulation Duration: <strong className="text-slate-200">{input.duration_h} hrs</strong> ({(input.duration_h / 24).toFixed(0)} days)
          </label>
          <input
            type="range"
            min="24"
            max="168"
            step="24"
            value={input.duration_h}
            onChange={(e) => updateField("duration_h", parseInt(e.target.value))}
            className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer mt-1"
          />
        </div>
      </div>

      {/* Bayesian TDM (MAP Forecasting) Lab Observations */}
      <div className="space-y-2 pt-2 border-t border-slate-800/80">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-teal-300 uppercase tracking-wider flex items-center space-x-1">
            <Activity className="w-3.5 h-3.5 text-teal-400" />
            <span>Bayesian TDM (MAP)</span>
          </span>
          <span className="text-[10px] text-slate-400">Optional Labs</span>
        </div>

        {input.tdm_observations.length > 0 && (
          <div className="space-y-1.5">
            {input.tdm_observations.map((obs, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between px-2.5 py-1 text-xs rounded-lg bg-teal-950/40 border border-teal-800/40 text-teal-200"
              >
                <span>t={obs.time_h}h &rarr; <strong>{obs.concentration} {input.drug === "Digoxin" ? "ng/mL" : "mcg/mL"}</strong></span>
                <button
                  type="button"
                  onClick={() => handleRemoveTDM(idx)}
                  className="text-slate-400 hover:text-rose-400"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center gap-1.5 text-xs">
          <input
            type="number"
            placeholder="Time (h)"
            value={newTdmTime}
            onChange={(e) => setNewTdmTime(e.target.value)}
            className="w-20 glass-input px-2 py-1 rounded-lg text-xs font-mono"
          />
          <input
            type="number"
            step="0.1"
            placeholder={`Obs Level (${input.drug === "Digoxin" ? "ng/mL" : "mcg/mL"})`}
            value={newTdmConc}
            onChange={(e) => setNewTdmConc(e.target.value)}
            className="flex-1 glass-input px-2 py-1 rounded-lg text-xs font-mono"
          />
          <button
            type="button"
            onClick={handleAddTDM}
            className="p-1.5 rounded-lg bg-teal-900/60 hover:bg-teal-800 border border-teal-700/50 text-teal-200"
            title="Add Blood Draw Level"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="pt-2 flex flex-col gap-2">
        <button
          type="button"
          onClick={onSimulate}
          disabled={isSimulating}
          className="w-full py-2.5 px-4 rounded-xl font-semibold text-xs text-slate-900 bg-gradient-to-r from-cyan-400 via-teal-400 to-emerald-400 hover:from-cyan-300 hover:to-emerald-300 shadow-lg shadow-cyan-950/60 transition-all flex items-center justify-center space-x-2 disabled:opacity-50"
        >
          {isSimulating ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Simulating Digital Twin...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Execute Digital Twin Simulation</span>
            </>
          )}
        </button>

        <button
          type="button"
          onClick={onRunTrial}
          className="w-full py-2 px-3 rounded-xl font-medium text-xs text-indigo-300 bg-indigo-950/60 hover:bg-indigo-900/60 border border-indigo-700/50 transition-colors flex items-center justify-center space-x-1.5"
        >
          <Activity className="w-3.5 h-3.5 text-indigo-400" />
          <span>Run In-Silico Population Trial</span>
        </button>
      </div>
    </aside>
  );
};
