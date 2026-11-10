"""
hetero_graph.py

Constructs a heterogeneous graph representation of patient physiology for use
with torch_geometric. Node types represent the patient, an administered drug,
and the major organ systems / monitoring subsystems involved in
pharmacokinetics and toxicity surveillance.

Node types
----------
Patient          : demographic / baseline physiological state
Drug              : the administered compound and its dosing profile
Liver             : hepatic function and metabolism state
Kidney            : renal function and elimination state
Cardiovascular    : cardiac / hemodynamic state
Safety_Monitor    : aggregated monitoring node used to read out risk signals

Edge types (directed, physiologically motivated)
--------------------------------------------------
(Patient, has_drug, Drug)
(Drug, cleared_by, Liver)              # hepatic clearance pathway
(Drug, eliminated_by, Kidney)          # renal elimination pathway
(Drug, affects, Cardiovascular)        # cardiotoxic / hemodynamic effect
(Liver, feeds_signal, Safety_Monitor)
(Kidney, feeds_signal, Safety_Monitor)
(Cardiovascular, feeds_signal, Safety_Monitor)
(Patient, baseline_state, Liver)
(Patient, baseline_state, Kidney)
(Patient, baseline_state, Cardiovascular)

Reverse edges are added automatically via `T.ToUndirected()` is intentionally
NOT used here (the physiology is directional), but explicit reverse relations
are defined where information should flow both ways (e.g. organ state
influences drug handling, and drug handling stresses the organ).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


NODE_TYPES: List[str] = [
    "Patient",
    "Drug",
    "Liver",
    "Kidney",
    "Cardiovascular",
    "Safety_Monitor",
]

# (src_type, relation, dst_type)
EDGE_TYPES: List[tuple] = [
    ("Patient", "has_drug", "Drug"),
    ("Drug", "cleared_by", "Liver"),
    ("Liver", "clears", "Drug"),
    ("Drug", "eliminated_by", "Kidney"),
    ("Kidney", "eliminates", "Drug"),
    ("Drug", "affects", "Cardiovascular"),
    ("Cardiovascular", "affected_by", "Drug"),
    ("Patient", "baseline_state", "Liver"),
    ("Patient", "baseline_state", "Kidney"),
    ("Patient", "baseline_state", "Cardiovascular"),
    ("Liver", "feeds_signal", "Safety_Monitor"),
    ("Kidney", "feeds_signal", "Safety_Monitor"),
    ("Cardiovascular", "feeds_signal", "Safety_Monitor"),
    ("Safety_Monitor", "monitors", "Liver"),
    ("Safety_Monitor", "monitors", "Kidney"),
    ("Safety_Monitor", "monitors", "Cardiovascular"),
]

# Expected raw feature dimensionality per node type. These are placeholders
# that should be aligned with the actual feature engineering pipeline
# (e.g. lab panels, dosing schedule, vitals).
DEFAULT_FEATURE_DIMS: Dict[str, int] = {
    "Patient": 12,          # age, sex, weight, BMI, comorbidity flags, ...
    "Drug": 8,               # dose, route, half-life, protein binding, ...
    "Liver": 6,              # ALT, AST, bilirubin, albumin, INR, ...
    "Kidney": 5,              # creatinine, eGFR, BUN, urine output, ...
    "Cardiovascular": 6,      # HR, BP (sys/dia), QTc, troponin, ...
    "Safety_Monitor": 4,      # aggregated alert counters / risk flags
}


@dataclass
class PatientPhysiologyRecord:
    """
    Container for a single patient-timepoint's raw feature vectors, keyed by
    node type. Each value is a 1D tensor of length DEFAULT_FEATURE_DIMS[node_type]
    (or a batch of such vectors, shape [num_nodes_of_type, feat_dim]).
    """

    features: Dict[str, Tensor] = field(default_factory=dict)

    def validate(self, feature_dims: Dict[str, int] = DEFAULT_FEATURE_DIMS) -> None:
        for node_type, expected_dim in feature_dims.items():
            if node_type not in self.features:
                raise ValueError(f"Missing features for node type '{node_type}'")
            actual_dim = self.features[node_type].shape[-1]
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Feature dim mismatch for '{node_type}': "
                    f"expected {expected_dim}, got {actual_dim}"
                )


def build_hetero_graph(
    record: PatientPhysiologyRecord,
    edge_index_overrides: Optional[Dict[tuple, Tensor]] = None,
) -> HeteroData:
    """
    Build a single-patient HeteroData graph instance.

    Parameters
    ----------
    record : PatientPhysiologyRecord
        Per-node-type raw feature tensors for this patient/timepoint.
    edge_index_overrides : dict, optional
        Mapping of (src, rel, dst) -> LongTensor[2, num_edges] to override the
        default single-node-per-type topology (useful when a patient has
        multiple drugs, or multiple historical organ-state snapshots).

    Returns
    -------
    HeteroData
        A torch_geometric HeteroData object ready for HeteroConv message passing.
    """
    record.validate()
    data = HeteroData()

    # --- Node features ---
    for node_type in NODE_TYPES:
        feats = record.features[node_type]
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)  # ensure shape [num_nodes, feat_dim]
        data[node_type].x = feats.float()

    # --- Edges ---
    # Default topology: one node per type (index 0), fully connected per
    # the physiological relations defined in EDGE_TYPES. Callers with
    # multiple nodes per type (e.g. several drugs) should pass
    # edge_index_overrides.
    for src, rel, dst in EDGE_TYPES:
        key = (src, rel, dst)
        if edge_index_overrides and key in edge_index_overrides:
            data[key].edge_index = edge_index_overrides[key]
        else:
            num_src = data[src].x.size(0)
            num_dst = data[dst].x.size(0)
            src_idx = torch.arange(num_src).repeat_interleave(num_dst)
            dst_idx = torch.arange(num_dst).repeat(num_src)
            data[key].edge_index = torch.stack([src_idx, dst_idx], dim=0)

    return data


def build_batch(records: List[PatientPhysiologyRecord]) -> List[HeteroData]:
    """Convenience helper to build a list of HeteroData graphs for batching
    with torch_geometric.loader.DataLoader."""
    return [build_hetero_graph(r) for r in records]


if __name__ == "__main__":
    # Minimal smoke test with synthetic data
    torch.manual_seed(0)
    synthetic = PatientPhysiologyRecord(
        features={
            node_type: torch.randn(dim)
            for node_type, dim in DEFAULT_FEATURE_DIMS.items()
        }
    )
    graph = build_hetero_graph(synthetic)
    print(graph)
    for node_type in NODE_TYPES:
        print(node_type, graph[node_type].x.shape)
    for edge_type in EDGE_TYPES:
        print(edge_type, graph[edge_type].edge_index.shape)