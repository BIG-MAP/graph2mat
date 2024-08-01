"""E3nn operations to compute edge matrix blocks.

In edge matrix blocks, you tipically will have, for each edge,
a different message coming from each atom in the edge. The edge block
will tipically not be symmetric, but it is common that.

.. math::
    B_{ij} = B_{ji}^T
"""

from e3nn import o3, nn
import torch

from typing import Tuple

from ._utils import tp_out_irreps_with_instructions

__all__ = [
    "E3nnSimpleEdgeBlock",
    "E3nnEdgeBlockNodeMix",
]


class E3nnSimpleEdgeBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        if isinstance(irreps_in, (o3.Irreps, str)):
            irreps_in = [irreps_in]

        self.tensor_products = torch.nn.ModuleList(
            [
                o3.FullyConnectedTensorProduct(
                    this_irreps_in, this_irreps_in, irreps_out
                )
                for this_irreps_in in irreps_in
            ]
        )

    def forward(
        self, **tuple_kwargs: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        assert len(tuple_kwargs) == len(
            self.tensor_products
        ), f"Number of input tuples ({len(tuple_kwargs)}) must match number of tensor square operations ({len(self.tensor_products)})."

        tensor_tuples = iter(tuple_kwargs.values())

        final_value = self.tensor_products[0](*next(tensor_tuples))
        for i, tensor_tuple in enumerate(tensor_tuples):
            final_value = final_value + self.tensor_products[i + 1](*tensor_tuple)

        return final_value


class E3nnEdgeBlockNodeMix(torch.nn.Module):
    _data_get_edge_args = ("edge_feats",)

    def __init__(
        self,
        edge_feats_irreps: o3.Irreps,
        edge_messages_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        # Network to reduce node representations to scalar features
        self.nodes_linear = o3.Linear(node_feats_irreps, edge_feats_irreps)

        # The weights of the tensor are produced by a fully connected neural network
        # that takes the scalar representations of nodes and edges as input
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            edge_messages_irreps,
            edge_messages_irreps,
            irreps_out,
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

        edge_tp_input_irreps = edge_feats_irreps * 3
        assert edge_tp_input_irreps.lmax == 0
        input_dim = edge_tp_input_irreps.num_irreps
        self.edge_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 2 * [128] + [self.edges_tp.weight_numel],
            torch.nn.SiLU(),
        )

        # The final output is produced by a linear layer
        self.output_linear = o3.Linear(irreps_mid, irreps_out)

    def forward(
        self,
        edge_feats: Tuple[torch.Tensor, torch.Tensor],
        edge_messages: Tuple[torch.Tensor, torch.Tensor],
        node_feats: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Convert nodes to scalar features
        scalar_node_feats_sender = self.nodes_linear(node_feats[0])
        scalar_node_feats_receiver = self.nodes_linear(node_feats[1])
        scalar_feats = torch.concatenate(
            (scalar_node_feats_sender, scalar_node_feats_receiver, edge_feats[0]), dim=1
        )
        # Obtain weights for edge tensor product
        edge_tp_weights = self.edge_tp_weights(scalar_feats)

        # Compute edge tensor product
        edges_tp = self.edges_tp(edge_messages[0], edge_messages[1], edge_tp_weights)

        # Compute final output
        output = self.output_linear(edges_tp)

        return output
