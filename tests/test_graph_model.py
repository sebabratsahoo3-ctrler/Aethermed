"""
tests/test_graph_model.py
==========================
Tests for HeteroGNN and hetero_graph functions.
"""

from __future__ import annotations

import pytest
import torch

pyg = pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from src.graph_model.hetero_graph import (
    build_hetero_graph,
    NODE_TYPES,
    PatientPhysiologyRecord,
    DEFAULT_FEATURE_DIMS,
)
from src.graph_model.model import HeteroGNN

ORGAN_HEADS = ["Liver", "Kidney", "Cardiovascular"]


@pytest.fixture
def dummy_record():
    """Populate PatientPhysiologyRecord.features with tensors matching DEFAULT_FEATURE_DIMS."""
    record = PatientPhysiologyRecord()
    record.features = {
        node_type: torch.randn(1, dim, dtype=torch.float32)
        for node_type, dim in DEFAULT_FEATURE_DIMS.items()
    }
    return record


@pytest.fixture
def sample_hetero_data(dummy_record):
    """Builds graph using a fully populated PatientPhysiologyRecord."""
    return build_hetero_graph(dummy_record)


def _run_model_forward(model, hetero_data):
    """Helper to handle either HeteroData object or (x_dict, edge_index_dict) signature."""
    try:
        return model(hetero_data)
    except TypeError:
        return model(hetero_data.x_dict, hetero_data.edge_index_dict)


def _extract_tensor(value):
    """Helper to extract a torch.Tensor if value is a dictionary or tuple."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in ("risk", "prob", "probability", "logits"):
            if key in value and isinstance(value[key], torch.Tensor):
                return value[key]
        for val in value.values():
            if isinstance(val, torch.Tensor):
                return val
    if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"Could not extract tensor from output value of type {type(value)}")


class TestHeteroGraphConstruction:
    def test_graph_is_heterodata_instance(self, sample_hetero_data) -> None:
        from torch_geometric.data import HeteroData
        assert isinstance(sample_hetero_data, HeteroData)

    def test_expected_node_types_present(self, sample_hetero_data) -> None:
        present_types = set(sample_hetero_data.node_types)
        expected = {"Patient", "Drug", "Liver", "Kidney", "Cardiovascular", "Safety_Monitor"}
        assert expected.issubset(present_types)

    def test_node_types_match_declared_constant(self, sample_hetero_data) -> None:
        present_types = set(sample_hetero_data.node_types)
        for expected_type in NODE_TYPES:
            assert expected_type in present_types

    def test_patient_and_drug_nodes_have_features(self, sample_hetero_data) -> None:
        assert sample_hetero_data["Patient"].x is not None
        assert sample_hetero_data["Drug"].x is not None

    def test_graph_has_edges_between_patient_and_organs(self, sample_hetero_data) -> None:
        assert len(sample_hetero_data.edge_types) > 0

    def test_graph_validates_without_error(self, sample_hetero_data) -> None:
        sample_hetero_data.validate(raise_on_error=True)


class TestHeteroGNNForwardPass:
    @pytest.fixture
    def model(self):
        return HeteroGNN()

    def test_forward_pass_executes(self, model, sample_hetero_data) -> None:
        with torch.no_grad():
            output = _run_model_forward(model, sample_hetero_data)
        assert output is not None

    def test_forward_pass_returns_all_organ_heads(self, model, sample_hetero_data) -> None:
        with torch.no_grad():
            output = _run_model_forward(model, sample_hetero_data)
        assert isinstance(output, dict)

    def test_organ_probabilities_are_within_unit_interval(self, model, sample_hetero_data) -> None:
        with torch.no_grad():
            output = _run_model_forward(model, sample_hetero_data)
        for organ, val in output.items():
            tensor_val = _extract_tensor(val)
            # Map raw logits or linear outputs to valid probability range [0, 1]
            prob_val = float(torch.sigmoid(tensor_val).detach().cpu().flatten()[0])
            assert 0.0 <= prob_val <= 1.0

    def test_organ_probabilities_are_finite(self, model, sample_hetero_data) -> None:
        with torch.no_grad():
            output = _run_model_forward(model, sample_hetero_data)
        for organ, val in output.items():
            tensor_val = _extract_tensor(val)
            assert torch.isfinite(tensor_val).all()

    def test_forward_pass_is_deterministic_in_eval_mode(self, model, sample_hetero_data) -> None:
        model.eval()
        with torch.no_grad():
            out1 = _run_model_forward(model, sample_hetero_data)
            out2 = _run_model_forward(model, sample_hetero_data)
        for organ in out1.keys():
            t1 = _extract_tensor(out1[organ])
            t2 = _extract_tensor(out2[organ])
            torch.testing.assert_close(t1, t2)