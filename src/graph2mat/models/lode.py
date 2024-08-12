import torch
from e3nn import o3

from graph2mat.bindings.salted import E3nnLODE
from graph2mat.bindings.e3nn import E3nnGraph2Mat


class MatrixE3nnLODE(torch.nn.Module):
    def __init__(self, lode: E3nnLODE, node_hidden_irreps, **kwargs):
        super().__init__()

        self.lode = lode

        self.linear = o3.Linear(lode.irreps_out, node_hidden_irreps)

        self.matrix_readout = E3nnGraph2Mat(
            irreps=dict(
                node_feats_irreps=node_hidden_irreps,
            ),
            **kwargs,
        )

    def forward(self, data):
        lode_node_feats = self.lode(data)
        node_feats = self.linear(lode_node_feats)
        node_labels, edge_labels = self.matrix_readout(data, node_feats=node_feats)

        return {
            "node_labels": node_labels,
            "edge_labels": edge_labels,
        }
