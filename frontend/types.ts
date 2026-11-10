export interface TDMPoint {
  time_h: number;
  concentration: number;
}

export interface PatientTwinInput {
  patient_id: string;
  age: number;
  gender: string;
  weight: number;
  serum_creatinine: number;
  egfr?: number;
  alt: number;
  ast: number;
  potassium: number;
  albumin: number;
  cyp2c9_status: string;
  vkorc1_status: string;
  comorbidities: string[];
  concomitant_drugs: string[];
  drug: string;
  dose_mg: number;
  interval_h: number;
  duration_h: number;
  loading_dose_mg?: number;
  tdm_observations: TDMPoint[];
}

export interface PKResponse {
  drug: string;
  display_unit: string;
  time_hours: number[];
  c_central_neural: number[];
  c_central_physics: number[];
  c_peripheral: number[];
  c_cql_recommended?: number[];
  cmax: number;
  cmin_trough: number;
  auc_24h: number;
  time_in_therapeutic_range_pct: number;
  half_life_hours: number;
  clearance_l_h: number;
  v_central_l: number;
  therapeutic_min: number;
  therapeutic_max: number;
  mass_conservation_error: number;
  active_model_source: string;
}

export interface OrganRiskItem {
  organ: string;
  risk_score: number;
  risk_band: "low" | "moderate" | "high";
  confidence: number;
  interval_lower: number;
  interval_upper: number;
}

export interface GraphNode {
  id: string;
  type: string;
  label: string;
  risk_score?: number;
  features: number[];
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  attention_weight: number;
}

export interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface RLResponse {
  baseline_dose_mg: number;
  cql_recommended_dose_mg: number;
  dose_delta_mg: number;
  analytical_target_dose_mg: number;
  recommendation_rationale: string;
  ope_metrics: {
    doubly_robust_return: number;
    weighted_importance_sampling_return: number;
    fitted_q_evaluation_return: number;
    behavior_baseline_return: number;
    pct_improvement_dr: number;
    effective_sample_size: number;
  };
}

export interface FeatureAttribution {
  feature: string;
  value: number;
  z_score: number;
  is_abnormal: boolean;
}

export interface SafetyOODResponse {
  mahalanobis_distance: number;
  threshold: number;
  is_ood: boolean;
  severity: "none" | "moderate" | "severe";
  feature_attributions: FeatureAttribution[];
  dosing_alerts: string[];
  is_dose_unsafe: boolean;
}

export interface CitationItem {
  citation_index: number;
  chunk_id: string;
  source_title: string;
  source_type: string;
  drug: string;
  similarity_score: number;
  citation: string;
  text_snippet: string;
  supported_claims: string[];
}

export interface RAGResponse {
  report_markdown: string;
  grounding_coverage_pct: number;
  executive_verdict: string;
  structured_citations: CitationItem[];
}

export interface SimulationResponse {
  patient_id: string;
  drug: string;
  pk_results: PKResponse;
  organ_risks: OrganRiskItem[];
  overall_risk: number;
  graph_topology: GraphPayload;
  rl_policy: RLResponse;
  safety_ood: SafetyOODResponse;
  rag_report: RAGResponse;
  bayesian_tdm_applied: boolean;
  bayesian_cl_multiplier: number;
  bayesian_v1_multiplier: number;
}

export interface CohortPatient {
  patient_id: string;
  age: number;
  gender: string;
  weight: number;
  serum_creatinine: number;
  egfr: number;
  alt: number;
  ast: number;
  cyp2c9_status: string;
  vkorc1_status: string;
  serum_potassium: number;
  albumin: number;
}

export interface DistributionMetric {
  counts: number[];
  bin_edges: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface CohortSummary {
  total_twins: number;
  mean_age: number;
  mean_weight: number;
  mean_creatinine: number;
  mean_egfr: number;
  mean_alt: number;
  mean_ast: number;
  distributions: Record<string, DistributionMetric>;
  correlation_matrix: Record<string, Record<string, number>>;
  sample_patients: CohortPatient[];
}

export interface TrialFilter {
  drug: string;
  standard_dose_mg: number;
  interval_h: number;
  min_age: number;
  max_age: number;
  max_egfr?: number;
  sample_size: number;
}

export interface PatientTrialOutcome {
  patient_id: string;
  age: number;
  egfr: number;
  standard_dose_mg: number;
  standard_ttr_pct: number;
  standard_trough: number;
  cql_dose_mg: number;
  cql_ttr_pct: number;
  cql_trough: number;
  clinical_outcome: string;
}

export interface TrialResponse {
  drug: string;
  sample_size: number;
  filter_description: string;
  standard_physician_regimen: string;
  aethermed_cql_regimen: string;
  standard_mean_ttr_pct: number;
  cql_mean_ttr_pct: number;
  ttr_gain_pct: number;
  standard_nephrotoxicity_incidence_pct: number;
  cql_nephrotoxicity_incidence_pct: number;
  toxicity_relative_reduction_pct: number;
  population_pk_summary: {
    time_hours: number[];
    standard_median: number[];
    standard_p10: number[];
    standard_p90: number[];
    cql_median: number[];
    cql_p10: number[];
    cql_p90: number[];
    therapeutic_min: number;
    therapeutic_max: number;
    display_unit: string;
  };
  patient_outcomes_sample: PatientTrialOutcome[];
}

export interface HealthResponse {
  status: string;
  platform: string;
  device: string;
  loaded_models: string[];
  chroma_chunk_count: number;
  cohort_twins_count: number;
}
