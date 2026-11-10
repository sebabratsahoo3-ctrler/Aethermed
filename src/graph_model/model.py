"""
model.py

Defines a heterogeneous graph neural network (`HeteroGNN`) that consumes the
graphs produced by `hetero_graph.build_hetero_graph` and predicts per-organ
toxicity risk scores.

Architecture
------------
- A per-node-type input projection (Linear) to a shared hidden dimension.
- Two (configurable) `HeteroConv` message-passing layers, each wrapping a
  per-edge-type GATConv or SAGEConv, aggregated with `sum`.
- A readout head applied to the organ node types (Liver, Kidney,
  Cardiovascular) that maps updated node embeddings to a toxicity risk logit
  per organ (plus an optional aggregated overall-risk head sourced from the
  Safety_Monitor node).

This model is a research/engineering scaffold for experimentation. It is not
a validated clinical decision-support tool, and its outputs should not be
used to guide patient care without independent clinical validation and
regulatory review.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv

from .hetero_graph import DEFAULT_FEATURE_DIMS, EDGE_TYPES, NODE_TYPES

ORGAN_NODE_TYPES: List[str] = ["Liver", "Kidney", "Cardiovascular"]


def _make_conv(conv_type: str, hidden_channels: int, heads: int = 2):
    """Instantiate a single conv layer for one edge type."""
    if conv_type == "gat":
        # concat=False averages head outputs so dims stay at hidden_channels
        return GATConv(
            (-1, -1),
            hidden_channels,
            heads=heads,
            concat=False,
            add_self_loops=False,
        )
    elif conv_type == "sage":
        return SAGEConv((-1, -1), hidden_channels)
    else:
        raise ValueError(f"Unknown conv_type '{conv_type}', expected 'gat' or 'sage'")


class HeteroGNN(nn.Module):
    """
    Heterogeneous GNN over the patient/drug/organ graph.

    Parameters
    ----------
    hidden_channels : int
        Hidden dimensionality used throughout the network.
    num_layers : int
        Number of stacked HeteroConv message-passing layers.
    conv_type : str
        'gat' for GATConv or 'sage' for SAGEConv per edge type.
    node_feature_dims : dict, optional
        Raw input feature dims per node type. Defaults to
        hetero_graph.DEFAULT_FEATURE_DIMS.
    edge_types : list, optional
        List of (src, rel, dst) triples. Defaults to hetero_graph.EDGE_TYPES.
    dropout : float
        Dropout applied after each conv layer.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 2,
        conv_type: str = "gat",
        node_feature_dims: Optional[Dict[str, int]] = None,
        edge_types: Optional[List[tuple]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_feature_dims = node_feature_dims or DEFAULT_FEATURE_DIMS
        self.edge_types = edge_types or EDGE_TYPES
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Per-node-type input projection to a shared hidden size.
        self.input_proj = nn.ModuleDict(
            {
                node_type: nn.Linear(dim, hidden_channels)
                for node_type, dim in self.node_feature_dims.items()
            }
        )

        # Stacked HeteroConv layers.
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_type: _make_conv(conv_type, hidden_channels)
                for edge_type in self.edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {nt: nn.LayerNorm(hidden_channels) for nt in NODE_TYPES}
                )
                for _ in range(num_layers)
            ]
        )

        # Per-organ toxicity risk head: hidden -> 1 logit.
        self.organ_risk_head = nn.ModuleDict(
            {
                organ: nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_channels // 2, 1),
                )
                for organ in ORGAN_NODE_TYPES
            }
        )

        # Aggregated overall risk head sourced from Safety_Monitor embedding.
        self.overall_risk_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def encode(
        self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[tuple, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Run input projection + message passing, return updated node embeddings."""
        h_dict = {
            node_type: F.relu(self.input_proj[node_type](x))
            for node_type, x in x_dict.items()
        }

        for conv, norm_dict in zip(self.convs, self.norms):
            out_dict = conv(h_dict, edge_index_dict)
            # HeteroConv only returns entries for node types with incoming edges;
            # fall back to the previous embedding for any node type untouched
            # in this layer (e.g. Patient may have no incoming edges).
            new_h_dict = {}
            for node_type in h_dict:
                if node_type in out_dict:
                    h = out_dict[node_type]
                    h = norm_dict[node_type](h)
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                else:
                    h = h_dict[node_type]
                new_h_dict[node_type] = h
            h_dict = new_h_dict

        return h_dict

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x_dict : dict[node_type -> Tensor[num_nodes, feat_dim]]
        edge_index_dict : dict[(src, rel, dst) -> LongTensor[2, num_edges]]

        Returns
        -------
        dict with:
            'organ_risk'   : dict[organ_name -> Tensor[num_nodes, 1] risk logits]
            'overall_risk' : Tensor[num_safety_monitor_nodes, 1] aggregated risk logit
            'embeddings'   : dict[node_type -> Tensor[num_nodes, hidden_channels]]
        """
        h_dict = self.encode(x_dict, edge_index_dict)

        organ_risk = {
            organ: self.organ_risk_head[organ](h_dict[organ])
            for organ in ORGAN_NODE_TYPES
            if organ in h_dict
        }

        overall_risk = None
        if "Safety_Monitor" in h_dict:
            overall_risk = self.overall_risk_head(h_dict["Safety_Monitor"])

        return {
            "organ_risk": organ_risk,
            "overall_risk": overall_risk,
            "embeddings": h_dict,
        }

    @staticmethod
    def risk_probabilities(
        output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Convert raw logits from `forward` into [0, 1] risk probabilities via sigmoid."""
        probs = {
            organ: torch.sigmoid(logits)
            for organ, logits in output["organ_risk"].items()
        }
        if output.get("overall_risk") is not None:
            probs["overall"] = torch.sigmoid(output["overall_risk"])
        return probs


if __name__ == "__main__":
    # Minimal smoke test wiring model.py to hetero_graph.py
    from .hetero_graph import PatientPhysiologyRecord, build_hetero_graph

    torch.manual_seed(0)
    record = PatientPhysiologyRecord(
        features={nt: torch.randn(dim) for nt, dim in DEFAULT_FEATURE_DIMS.items()}
    )
    graph = build_hetero_graph(record)

    model = HeteroGNN(hidden_channels=32, num_layers=2, conv_type="gat")
    out = model(graph.x_dict, graph.edge_index_dict)
    probs = HeteroGNN.risk_probabilities(out)

    for organ, p in probs.items():
        print(f"{organ} risk probability: {p.squeeze().tolist()}")