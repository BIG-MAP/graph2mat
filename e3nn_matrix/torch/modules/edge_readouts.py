from abc import ABC, abstractmethod

from e3nn import o3
import torch
from typing import Tuple

__all__ = [
    "EdgeBlock",
    "SimpleEdgeBlock",
    "SimpleEdgeBlockWithNodes"
]

class EdgeBlock(torch.nn.Module, ABC):
    """Base class for computing edge blocks of an orbital matrix.
    Parameters
    -----------
    edge_feats_irreps: o3.Irreps
    node_feats_irreps: o3.Irreps
    irreps_out: o3.Irreps
    """
    @abstractmethod
    def forward(self, 
        edge_feats: Tuple[torch.Tensor, torch.Tensor],
        edge_messages: Tuple[torch.Tensor, torch.Tensor],
        edge_index: Tuple[torch.Tensor, torch.Tensor],
        node_feats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return edge_feats[0]

class SimpleEdgeBlock(EdgeBlock):
    def __init__(self, edge_feats_irreps: o3.Irreps, node_feats_irreps: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        self.tp = o3.FullyConnectedTensorProduct(edge_feats_irreps, edge_feats_irreps, irreps_out)

    def forward(self, 
        edge_feats: Tuple[torch.Tensor, torch.Tensor],
        edge_messages: Tuple[torch.Tensor, torch.Tensor],
        edge_index: Tuple[torch.Tensor, torch.Tensor],
        node_feats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return self.tp(edge_messages[0], edge_messages[1])

class SimpleEdgeBlockWithNodes(EdgeBlock):
    def __init__(self, edge_feats_irreps: o3.Irreps, node_feats_irreps: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        self.nodes_tp = o3.FullyConnectedTensorProduct(node_feats_irreps, node_feats_irreps, irreps_out)

        self.edges_tp = o3.FullyConnectedTensorProduct(edge_feats_irreps, edge_feats_irreps, irreps_out)

    def forward(self, 
        edge_feats: Tuple[torch.Tensor, torch.Tensor],
        edge_messages: Tuple[torch.Tensor, torch.Tensor],
        edge_index: Tuple[torch.Tensor, torch.Tensor],
        node_feats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return self.nodes_tp(node_feats[0], node_feats[1]) + self.edges_tp(edge_messages[0], edge_messages[1])
