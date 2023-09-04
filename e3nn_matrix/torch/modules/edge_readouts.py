from abc import ABC, abstractmethod

from e3nn import o3, nn
import torch
from typing import Tuple

from mace.modules.irreps_tools import tp_out_irreps_with_instructions

from ._symm_product import FullyConnectedSymmTensorProduct

__all__ = [
    "EdgeBlock",
    "SimpleEdgeBlock",
    "SimpleEdgeBlockWithNodes",
    "EdgeBlockNodeMix",
]

class EdgeBlock(torch.nn.Module, ABC):
    """Base class for computing edge blocks of a basis-basis matrix.
    Parameters
    -----------
    edge_feats_irreps: o3.Irreps
    edge_messages_irreps: o3.Irreps
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

class SymmTransposeEdgeBlock(EdgeBlock):
    """Base class for computing edge blocks of a basis-basis matrix.
    This block supports transpose symmetric blocks, i.e. f(x, y) == f(y, x).T when symm_transpose==True.
    Parameters
    -----------
    edge_feats_irreps: o3.Irreps
    edge_messages_irreps: o3.Irreps
    node_feats_irreps: o3.Irreps
    irreps_out: o3.Irreps
    symm_transpose: bool = True
    """

class SimpleEdgeBlock(SymmTransposeEdgeBlock):
    def __init__(self, edge_feats_irreps: o3.Irreps, edge_messages_irreps: o3.Irreps, node_feats_irreps: o3.Irreps, irreps_out: o3.Irreps, symm_transpose: bool = True):
        super().__init__()

        self.symm_transpose = symm_transpose

        tp_class = FullyConnectedSymmTensorProduct if symm_transpose else o3.FullyConnectedTensorProduct

        self.tp = tp_class(edge_messages_irreps, edge_messages_irreps, irreps_out)

    def forward(self, 
        edge_feats: Tuple[torch.Tensor, torch.Tensor],
        edge_messages: Tuple[torch.Tensor, torch.Tensor],
        edge_index: Tuple[torch.Tensor, torch.Tensor],
        node_feats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return self.tp(edge_messages[0], edge_messages[1])

class SimpleEdgeBlockWithNodes(SymmTransposeEdgeBlock):
    def __init__(self, edge_feats_irreps: o3.Irreps, edge_messages_irreps: o3.Irreps, node_feats_irreps: o3.Irreps, irreps_out: o3.Irreps, symm_transpose: bool = True):
        super().__init__()

        self.symm_transpose = symm_transpose

        tp_class = FullyConnectedSymmTensorProduct if symm_transpose else o3.FullyConnectedTensorProduct

        self.nodes_tp = tp_class(node_feats_irreps, node_feats_irreps, irreps_out)

        self.edges_tp = tp_class(edge_messages_irreps, edge_messages_irreps, irreps_out)

    def forward(self, 
        edge_feats: Tuple[torch.Tensor, torch.Tensor],
        edge_messages: Tuple[torch.Tensor, torch.Tensor],
        edge_index: Tuple[torch.Tensor, torch.Tensor],
        node_feats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return self.nodes_tp(node_feats[0], node_feats[1]) + self.edges_tp(edge_messages[0], edge_messages[1])

class EdgeBlockNodeMix(EdgeBlock):
    def __init__(self, edge_feats_irreps: o3.Irreps, edge_messages_irreps: o3.Irreps, node_feats_irreps: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        # Network to reduce node representations to scalar features
        self.nodes_linear = o3.Linear(node_feats_irreps, edge_feats_irreps)

        # The weights of the tensor are produced by a fully connected neural network
        # that takes the scalar representations of nodes and edges as input
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            edge_messages_irreps, edge_messages_irreps, irreps_out,
        )
        # Tensor product between edge features from sender and receiver
        self.edges_tp = o3.TensorProduct(
            edge_messages_irreps,
            edge_messages_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        irreps_mid = irreps_mid.simplify()

        edge_tp_input_irreps = edge_feats_irreps*3
        assert edge_tp_input_irreps.lmax == 0
        input_dim = edge_tp_input_irreps.num_irreps
        self.edge_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 2 * [128] + [self.edges_tp.weight_numel], torch.nn.SiLU(),
        )

        # The final output is produced by a linear layer
        self.output_linear = o3.Linear(irreps_mid, irreps_out)

    def forward(self, 
        edge_feats: Tuple[torch.Tensor, torch.Tensor],
        edge_messages: Tuple[torch.Tensor, torch.Tensor],
        edge_index: Tuple[torch.Tensor, torch.Tensor],
        node_feats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # Convert nodes to scalar features
        scalar_node_feats_sender = self.nodes_linear(node_feats[0])
        scalar_node_feats_receiver = self.nodes_linear(node_feats[1])
        scalar_feats = torch.concatenate((scalar_node_feats_sender, scalar_node_feats_receiver, edge_feats[0]) ,dim=1)
        # Obtain weights for edge tensor product
        edge_tp_weights = self.edge_tp_weights(scalar_feats)

        # Compute edge tensor product
        edges_tp = self.edges_tp(edge_messages[0], edge_messages[1], edge_tp_weights)

        # Compute final output
        output = self.output_linear(edges_tp)

        return output
