"use client";

import React, { useState, useEffect } from "react";
import { Header } from "@/components/Header";
import { CockpitSidebar } from "@/components/CockpitSidebar";
import { PKPDView } from "@/components/PKPDView";
import { OrganRiskView } from "@/components/OrganRiskView";
import { RLPolicyView } from "@/components/RLPolicyView";
import { ClinicalTrialView } from "@/components/ClinicalTrialView";
import { RAGConsultView } from "@/components/RAGConsultView";
import {
  Activity,
  ShieldAlert,
  Compass,
  Users,
  BookOpen,
  Sparkles,
} from "lucide-react";
import {
  PatientTwinInput,
  SimulationResponse,
  HealthResponse,
  CohortSummary,
  TrialResponse,
  TrialFilter,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [cohortSummary, setCohortSummary] = useState<CohortSummary | null>(null);
  const [simulation, setSimulation] = useState<SimulationResponse | null>(null);
  const [trialData, setTrialData] = useState<TrialResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"pk" | "organ" | "rl" | "trial" | "rag">("pk");
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [isLoadingTrial, setIsLoadingTrial] = useState<boolean>(false);

  // Digital Twin Input State
  const [input, setInput] = useState<PatientTwinInput>({
    patient_id: "TWIN-8821",
    age: 65,
    gender: "female",
    weight: 72,
    serum_creatinine: 1.2,
    egfr: 65.0,
    alt: 35,
    ast: 38,
    potassium: 4.25,
    albumin: 4.1,
    cyp2c9_status: "*1/*1",
    vkorc1_status: "G/A",
    comorbidities: ["hypertension"],
    concomitant_drugs: ["amiodarone"],
    drug: "Warfarin",
    dose_mg: 5.0,
    interval_h: 24,
    duration_h: 72,
    tdm_observations: [],
  });

  // Fetch initial telemetry and cohort data on mount
  useEffect(() => {
    const fetchInit = async () => {
      try {
        const healthRes = await fetch(`${API_BASE}/api/health`);
        if (healthRes.ok) {
          const hData = await healthRes.json();
          setHealth(hData);
        }

        const cohortRes = await fetch(`${API_BASE}/api/cohort?limit=25`);
        if (cohortRes.ok) {
          const cData = await cohortRes.json();
          setCohortSummary(cData);
        }
      } catch (err) {
        console.warn("Backend connection pending:", err);
      }
    };

    fetchInit();
  }, []);

  // Run initial simulation once backend is detected or on mount
  useEffect(() => {
    runSimulation(input);
  }, []);

  const runSimulation = async (payload: PatientTwinInput) => {
    setIsSimulating(true);
    try {
      const res = await fetch(`${API_BASE}/api/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data: SimulationResponse = await res.json();
        setSimulation(data);
      } else {
        console.error("Simulation failed:", await res.text());
      }
    } catch (err) {
      console.error("Error connecting to simulation endpoint:", err);
    } finally {
      setIsSimulating(false);
    }
  };

  const handleRunTrial = async (trialFilter: TrialFilter) => {
    setIsLoadingTrial(true);
    try {
      const res = await fetch(`${API_BASE}/api/trial/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(trialFilter),
      });

      if (res.ok) {
        const data: TrialResponse = await res.json();
        setTrialData(data);
        setActiveTab("trial");
      }
    } catch (err) {
      console.error("Trial execution failed:", err);
    } finally {
      setIsLoadingTrial(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#030712] text-slate-100">
      {/* Telemetry Header */}
      <Header health={health} safety={simulation?.safety_ood || null} />

      {/* Main Clinical Twin Cockpit */}
      <main className="flex-1 max-w-[1720px] w-full mx-auto p-4 sm:p-6 flex flex-col lg:flex-row gap-6">
        {/* Left: Patient Digital Twin Sidebar */}
        <CockpitSidebar
          input={input}
          onChange={setInput}
          onSimulate={() => runSimulation(input)}
          onRunTrial={() => {
            setActiveTab("trial");
            handleRunTrial({
              drug: input.drug,
              standard_dose_mg: input.dose_mg,
              interval_h: input.interval_h,
              min_age: Math.max(18, input.age - 15),
              max_age: Math.min(95, input.age + 15),
              max_egfr: input.egfr || 60,
              sample_size: 25,
            });
          }}
          isSimulating={isSimulating}
          sampleCohort={cohortSummary?.sample_patients || []}
        />

        {/* Right: Tabbed Clinical Workspace */}
        <div className="flex-1 flex flex-col gap-5 min-w-0">
          {/* Navigation Tabs Bar */}
          <nav className="glass-panel p-1.5 rounded-2xl border border-slate-800 flex flex-wrap items-center gap-1.5 shadow-lg">
            <button
              type="button"
              onClick={() => setActiveTab("pk")}
              className={`flex items-center space-x-2 py-2 px-4 rounded-xl text-xs font-semibold transition-all ${
                activeTab === "pk"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-950/60"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
              }`}
            >
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>PK/PD Simulation Studio</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("organ")}
              className={`flex items-center space-x-2 py-2 px-4 rounded-xl text-xs font-semibold transition-all ${
                activeTab === "organ"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm shadow-rose-950/60"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
              }`}
            >
              <ShieldAlert className="w-4 h-4 text-rose-400" />
              <span>Organ Risk &amp; GNN Topology</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("rl")}
              className={`flex items-center space-x-2 py-2 px-4 rounded-xl text-xs font-semibold transition-all ${
                activeTab === "rl"
                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm shadow-purple-950/60"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
              }`}
            >
              <Compass className="w-4 h-4 text-purple-400" />
              <span>Offline RL (CQL) Dosing Policy</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("trial")}
              className={`flex items-center space-x-2 py-2 px-4 rounded-xl text-xs font-semibold transition-all ${
                activeTab === "trial"
                  ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 shadow-sm shadow-indigo-950/60"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
              }`}
            >
              <Users className="w-4 h-4 text-indigo-400" />
              <span>In-Silico Virtual Trial</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("rag")}
              className={`flex items-center space-x-2 py-2 px-4 rounded-xl text-xs font-semibold transition-all ${
                activeTab === "rag"
                  ? "bg-teal-500/20 text-teal-300 border border-teal-500/40 shadow-sm shadow-teal-950/60"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
              }`}
            >
              <BookOpen className="w-4 h-4 text-teal-400" />
              <span>Literature RAG Consult &amp; Citations</span>
            </button>
          </nav>

          {/* Active View Container */}
          <div className="flex-1">
            {activeTab === "pk" && <PKPDView pk={simulation?.pk_results || null} />}

            {activeTab === "organ" && (
              <OrganRiskView
                organRisks={simulation?.organ_risks || []}
                overallRisk={simulation?.overall_risk || 0.25}
                graphTopology={simulation?.graph_topology || null}
              />
            )}

            {activeTab === "rl" && (
              <RLPolicyView rl={simulation?.rl_policy || null} drug={input.drug} />
            )}

            {activeTab === "trial" && (
              <ClinicalTrialView
                trialData={trialData}
                onExecuteTrial={handleRunTrial}
                isLoading={isLoadingTrial}
              />
            )}

            {activeTab === "rag" && (
              <RAGConsultView rag={simulation?.rag_report || null} drug={input.drug} />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
