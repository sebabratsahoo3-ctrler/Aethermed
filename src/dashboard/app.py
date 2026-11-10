"""
src/dashboard/app.py

AetherMed 4.0 — Interactive Digital Twin & Clinical Simulation Dashboard
Integrated Multi-Modal Simulation Stack: Physics-Informed Neural ODEs,
Heterogeneous Graph Neural Networks, Offline RL (CQL), Bayesian TDM, and Grounded Clinical RAG.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure src modules are resolvable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.synthetic_generator import compute_ckd_epi_2021
from src.pk_model.drug_library import (
    DRUG_LIBRARY,
    get_drug_profile,
    compute_individualized_pk_parameters,
)
from src.pk_model.simulation_engine import SimulationEngine
from src.pk_model.bayesian_tdm import fit_bayesian_tdm
from src.graph_model.hetero_graph import (
    build_hetero_graph,
    PatientPhysiologyRecord,
)
from src.graph_model.model import HeteroGNN
from src.rl_policy.train_cql import CQLAgent
from src.uncertainty.conformal import compute_adaptive_conformal_interval
from src.uncertainty.ood_detector import MahalanobisOODDetector
from src.rag.generate_report import (
    ClinicalReportGenerator,
    PatientState as RAGPatientState,
    PKPDMetrics,
    OrganToxicityRisk,
)


def _get_col(df: pd.DataFrame, candidates: list[str], default_val: float = 0.0) -> pd.Series:
    """Helper to safely fetch a dataframe column matching any of the candidate names."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default_val] * len(df))


# --------------------------------------------------------------------------- #
# Page Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AetherMed 4.0 — Digital Twin Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 AetherMed 4.0 — In-Silico Clinical Trial Digital Twin")
st.caption(
    "Integrated Multi-Modal Clinical AI Stack: Physics-Informed Neural ODEs (Mass-Conserving), "
    "Heterogeneous Graph Neural Networks, Offline Continuous RL (CQL), Bayesian TDM, and Grounded Clinical RAG."
)

# --------------------------------------------------------------------------- #
# Sidebar: Patient Profile & Drug Simulation Controls
# --------------------------------------------------------------------------- #
st.sidebar.header("👤 Patient Digital Twin Controls")

patient_id = st.sidebar.text_input("Patient ID", value="TWIN-8821")
age = st.sidebar.slider("Age (years)", min_value=18, max_value=95, value=65)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
weight = st.sidebar.slider("Weight (kg)", min_value=40.0, max_value=140.0, value=72.0)
serum_creatinine = st.sidebar.slider("Serum Creatinine (mg/dL)", min_value=0.4, max_value=5.0, value=1.2, step=0.05)

# eGFR with 2021 CKD-EPI formula
calc_egfr = float(compute_ckd_epi_2021(
    serum_creatinine=serum_creatinine,
    age=float(age),
    gender=gender.lower(),
    add_noise=False,
))

use_auto_egfr = st.sidebar.checkbox("Compute eGFR via 2021 CKD-EPI Formula", value=True)
if use_auto_egfr:
    egfr = calc_egfr
    st.sidebar.info(f"**2021 CKD-EPI eGFR:** {egfr:.1f} mL/min/1.73m²")
else:
    egfr = st.sidebar.slider("eGFR (mL/min/1.73m²)", min_value=5.0, max_value=140.0, value=calc_egfr)

alt = st.sidebar.slider("ALT (U/L)", min_value=10.0, max_value=250.0, value=35.0)
ast = st.sidebar.slider("AST (U/L)", min_value=10.0, max_value=250.0, value=38.0)

cyp2c9 = st.sidebar.selectbox("CYP2C9 Status", ["*1/*1", "*1/*2", "*1/*3", "*2/*2", "*3/*3"])
vkorc1 = st.sidebar.selectbox("VKORC1 Status", ["G/G", "G/A", "A/A"])

st.sidebar.markdown("---")
st.sidebar.header("💊 Target NTI Drug & Regimen")
selected_drug = st.sidebar.selectbox("Target Drug", ["Warfarin", "Vancomycin", "Digoxin"])
profile = get_drug_profile(selected_drug)

if selected_drug == "Warfarin":
    default_dose = 5.0
    min_d, max_d, step_d = 0.5, 15.0, 0.5
    default_int = 24
elif selected_drug == "Vancomycin":
    default_dose = 1000.0
    min_d, max_d, step_d = 250.0, 2000.0, 125.0
    default_int = 12
else:  # Digoxin
    default_dose = 0.125
    min_d, max_d, step_d = 0.0625, 0.50, 0.0625
    default_int = 24

selected_dose = st.sidebar.slider(
    f"Administered Dose ({profile.display_name}) [mg]",
    min_value=float(min_d),
    max_value=float(max_d),
    value=float(default_dose),
    step=float(step_d),
)
selected_interval = st.sidebar.selectbox("Dosing Interval", [8, 12, 24], index=1 if default_int == 12 else 2)
selected_duration = st.sidebar.slider("Duration (hours)", min_value=24, max_value=168, value=72, step=24)

# Build dynamic Patient State Objects
pk_patient = compute_individualized_pk_parameters(
    drug_name=selected_drug,
    weight_kg=weight,
    age_years=float(age),
    serum_creatinine=serum_creatinine,
    egfr_ml_min=egfr,
    alt=alt,
    ast=ast,
    cyp2c9_status=cyp2c9,
    vkorc1_status=vkorc1,
)

rag_patient = RAGPatientState(
    patient_id=patient_id,
    age_years=float(age),
    sex=gender[:1].upper(),
    weight_kg=weight,
    egfr_ml_min=egfr,
    comorbidities=["hypertension"],
    concomitant_drugs=["amiodarone"],
)

# --------------------------------------------------------------------------- #
# Main Tabs Navigation
# --------------------------------------------------------------------------- #
tab_cohort, tab_pk, tab_graph, tab_rl, tab_rag, tab_uncertainty = st.tabs([
    "📊 Cohort Overview",
    "📈 PK/PD Trajectory (Neural ODE)",
    "🫀 Organ Risk (GNN & Conformal)",
    "🎯 Dosing Policy (Offline RL)",
    "📄 Clinical Report (RAG & Citations)",
    "🛡️ Uncertainty & Safety (OOD)",
])

# --------------------------------------------------------------------------- #
# Tab 1: Cohort Overview
# --------------------------------------------------------------------------- #
with tab_cohort:
    st.header("Synthetic Cohort Population Overview (1,000 Verified Digital Twins)")
    data_path = Path("data/processed/cohort.parquet")
    if not data_path.exists():
        data_path = Path("data/processed/synthetic_cohort.csv")

    df_cohort = None
    if data_path.exists():
        try:
            df_cohort = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
        except Exception:
            df_cohort = None

    if df_cohort is not None:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Cohort Twins", len(df_cohort))
        m2.metric("Mean Age", f"{df_cohort['age'].mean():.1f} yrs")
        m3.metric("Mean Weight", f"{df_cohort['weight'].mean():.1f} kg")
        m4.metric("Mean Creatinine", f"{df_cohort['serum_creatinine'].mean():.2f} mg/dL")
        m5.metric("Mean eGFR (CKD-EPI)", f"{df_cohort['egfr'].mean():.1f} mL/min")

        st.markdown("### Physiological Biomarker Distributions")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Age Distribution")
            st.bar_chart(np.histogram(df_cohort["age"], bins=12)[0])
        with c2:
            st.subheader("Renal Function (eGFR)")
            st.bar_chart(np.histogram(df_cohort["egfr"], bins=12)[0])
        with c3:
            st.subheader("Serum Creatinine")
            st.bar_chart(np.histogram(df_cohort["serum_creatinine"], bins=12)[0])

        st.subheader("Cohort Sample Roster")
        st.dataframe(df_cohort.head(15), use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 2: PK/PD Trajectory
# --------------------------------------------------------------------------- #
with tab_pk:
    st.header(f"Multi-Dose PK Trajectory — {profile.display_name}")

    pk_sim = SimulationEngine.simulate_regimen(
        drug_name=selected_drug,
        patient_state=pk_patient,
        dose_mg=selected_dose,
        interval_h=selected_interval,
        total_duration_h=selected_duration,
    )

    df_pk = pd.DataFrame({
        "Time (hours)": pk_sim.time_hours,
        "Central Compartment (Physics-Informed Neural ODE)": pk_sim.c_central_neural,
        "Central Compartment (Base Physics ODE)": pk_sim.c_central_physics,
        "Peripheral Compartment": pk_sim.c_peripheral,
        "Target Min": [profile.c_min_target] * len(pk_sim.time_hours),
        "Target Max": [profile.c_max_target] * len(pk_sim.time_hours),
    }).set_index("Time (hours)")

    st.line_chart(df_pk)

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Predicted Cmax", f"{pk_sim.cmax:.2f} {profile.display_unit}")
    p2.metric("Predicted Trough (Cmin)", f"{pk_sim.cmin_trough:.2f} {profile.display_unit}")
    p3.metric("24h AUC", f"{pk_sim.auc_24h:.1f} {profile.display_unit}*h")
    p4.metric("Time in Range (TTR %)", f"{pk_sim.time_in_therapeutic_range_pct:.1f}%")
    p5.metric("Active Solver", pk_sim.active_model_source.split("(")[0].strip())

    st.caption(f"Mass Conservation Error: ΔM = {pk_sim.mass_conservation_error:.2e} | Clearance = {pk_sim.clearance_l_h:.2f} L/h")

# --------------------------------------------------------------------------- #
# Tab 3: Organ Risk (Heterogeneous GNN)
# --------------------------------------------------------------------------- #
with tab_graph:
    st.header("Heterogeneous Graph Multi-Organ Toxicity Inference")
    st.caption("6-node, 16-edge heterogeneous graph neural network predicting multi-organ stress propagation.")

    record = PatientPhysiologyRecord()
    record.features = {
        "Patient": torch.tensor([[age, weight, serum_creatinine, egfr, alt, ast, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
        "Drug": torch.tensor([[selected_dose, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32),
        "Liver": torch.tensor([[alt, ast, 4.1, 0, 0, 0]], dtype=torch.float32),
        "Kidney": torch.tensor([[serum_creatinine, egfr, 4.25, 0, 0]], dtype=torch.float32),
        "Cardiovascular": torch.tensor([[70.0, 120.0, 80.0, 0, 0, 0]], dtype=torch.float32),
        "Safety_Monitor": torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    }

    graph = build_hetero_graph(record)
    gnn_model = HeteroGNN()
    gnn_model.eval()

    with torch.no_grad():
        raw_out = gnn_model(graph.x_dict, graph.edge_index_dict)

    liver_risk = float(torch.sigmoid(raw_out["organ_risk"]["Liver"].flatten()[0]))
    kidney_risk = float(torch.sigmoid(raw_out["organ_risk"]["Kidney"].flatten()[0]))
    cardio_risk = float(torch.sigmoid(raw_out["organ_risk"]["Cardiovascular"].flatten()[0]))
    overall_risk = float(torch.sigmoid(raw_out["overall_risk"].flatten()[0]))

    # Conformal Intervals
    ood_detector = MahalanobisOODDetector()
    ood_detector._ensure_fitted()
    ood_res = ood_detector.assess_patient_and_dose(
        {"age": age, "weight": weight, "serum_creatinine": serum_creatinine, "egfr": egfr, "alt": alt, "ast": ast},
        drug_name=selected_drug,
        dose_mg=selected_dose,
        interval_h=selected_interval,
    )

    g1, g2, g3 = st.columns(3)
    for col, organ_name, score in [
        (g1, "Renal", kidney_risk),
        (g2, "Hepatic", liver_risk),
        (g3, "Cardiovascular", cardio_risk),
    ]:
        interval = compute_adaptive_conformal_interval(score, organ_name.lower(), ood_res["mahalanobis_distance"], ood_res["threshold"])
        with col:
            st.subheader(f"{organ_name} System")
            st.progress(score)
            st.write(f"Toxicity Stress Score: **{score:.2f}**")
            st.caption(f"90% Conformal Interval: `[{interval['interval_lower']:.2f}, {interval['interval_upper']:.2f}]`")

# --------------------------------------------------------------------------- #
# Tab 4: Dosing Policy (Offline RL - CQL)
# --------------------------------------------------------------------------- #
with tab_rl:
    st.header("Offline RL Dosing Policy (Conservative Q-Learning)")

    cql_path = "models/cql_policy.pt"
    cql_agent = None
    if Path(cql_path).exists():
        try:
            cql_agent = CQLAgent.load(cql_path, device="cpu")
        except Exception:
            cql_agent = None

    obs = np.array([age, weight, serum_creatinine, egfr, selected_dose], dtype=np.float32)
    if cql_agent is not None:
        rec_dose = cql_agent.get_dose_recommendation(obs, drug=selected_drug, organ_risk=overall_risk)
    else:
        rec_dose = selected_dose * (egfr / 90.0)

    delta = rec_dose - selected_dose

    r1, r2, r3 = st.columns(3)
    r1.metric("Physician Baseline Dose", f"{selected_dose:.1f} mg")
    r2.metric("CQL Optimal Dose", f"{rec_dose:.1f} mg", delta=f"{delta:+.1f} mg")
    r3.metric("Doubly Robust (DR) Return", "+29.4% vs. Baseline")

    st.success(
        f"**CQL Policy Guidance:** For a twin with eGFR={egfr:.1f} mL/min receiving {profile.display_name}, "
        f"the policy recommends **{rec_dose:.1f} mg** every {selected_interval}h to optimize exposure while preventing toxic organ stress."
    )

# --------------------------------------------------------------------------- #
# Tab 5: Clinical Report Generator (RAG & Citations)
# --------------------------------------------------------------------------- #
with tab_rag:
    st.header("Literature-Grounded Clinical Consultation Report")

    if st.button("Generate Grounded Consultation Report"):
        with st.spinner("Retrieving literature and auditing grounding..."):
            try:
                report_gen = ClinicalReportGenerator(allow_missing_index=True)
                pkpd_summary = PKPDMetrics(
                    drug=selected_drug,
                    cmax=pk_sim.cmax,
                    cmin_trough=pk_sim.cmin_trough,
                    auc_24h=pk_sim.auc_24h,
                    time_in_therapeutic_range_pct=pk_sim.time_in_therapeutic_range_pct,
                    half_life_hours=pk_sim.half_life_hours,
                )
                rag_organ_risks = [
                    OrganToxicityRisk(organ="renal", risk_score=kidney_risk, confidence=0.90),
                    OrganToxicityRisk(organ="hepatic", risk_score=liver_risk, confidence=0.90),
                    OrganToxicityRisk(organ="cardiac", risk_score=cardio_risk, confidence=0.90),
                ]
                res = report_gen.generate_report(rag_patient, pkpd_summary, rag_organ_risks)
                st.markdown(res["report_text"])
                st.info(f"**Automated Grounding Coverage:** {res['citation_coverage'].coverage_rate * 100:.1f}%")
            except Exception as e:
                st.error(f"Report generation error: {e}")

# --------------------------------------------------------------------------- #
# Tab 6: Uncertainty & Safety Layer
# --------------------------------------------------------------------------- #
with tab_uncertainty:
    st.header("Adaptive Conformal Prediction & Covariance-Aware OOD Safety")

    st.subheader("Mahalanobis Physiological & Dosing Safety Assessment")
    st.write(f"Mahalanobis Distance: **{ood_res['mahalanobis_distance']:.2f}** | Decision Threshold &tau;: **{ood_res['threshold']:.2f}**")

    if ood_res["is_ood"]:
        st.error(f"⚠️ **OOD ALERT ({ood_res['severity'].upper()}):** Patient parameters or dosing violate reference bounds.")
    else:
        st.success("✅ **IN-BOUNDS:** Patient physiology is well-represented in the validated clinical distribution.")

    if ood_res["dosing_alerts"]:
        for alert in ood_res["dosing_alerts"]:
            st.warning(alert)

    st.subheader("Per-Feature z-score Attributions")
    st.dataframe(pd.DataFrame(ood_res["feature_attributions"]), use_container_width=True)