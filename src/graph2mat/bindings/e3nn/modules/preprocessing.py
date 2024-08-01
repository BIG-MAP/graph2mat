from typing import Tuple, Union, Dict

import torch
from e3nn import o3, nn

from graph2mat.bindings.torch import TorchBasisMatrixData
from ._utils import tp_out_irreps_with_instructions, scatter_sum

__all__ = [
    "E3nnInteraction",
    "E3nnEdgeMessageBlock",
]


class E3nnInteraction(torch.nn.Module):
    """Basically MACE's RealAgnosticResidualInteractionBlock, without reshapes.

    This function takes a graph and returns new states for the nodes.

    This function can be used for the preprocessing step of both nodes and edges.
    """

    def __init__(
        self,
        irreps: Dict[str, o3.Irreps],
        avg_num_neighbors: float = 10,
    ) -> None:
        super().__init__()

        node_feats_irreps = irreps["node_feats_irreps"]
        # node_attrs_irreps = irreps["node_attrs_irreps"]
        edge_attrs_irreps = irreps["edge_attrs_irreps"]
        edge_feats_irreps = irreps["edge_feats_irreps"]
        target_irreps = irreps["node_feats_irreps"]
        # hidden_irreps = irreps["node_feats_irreps"]

        # First linear
        self.linear_up = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            edge_attrs_irreps,
            target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            node_feats_irreps,
            edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.SiLU(),
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_mji = irreps_mid
        self.linear = o3.Linear(
            irreps_mid, target_irreps, internal_weights=True, shared_weights=True
        )

        self.avg_num_neighbors = avg_num_neighbors

        self.irreps_out = target_irreps

    def forward(
        self,
        data: TorchBasisMatrixData,
        node_feats: torch.Tensor,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        edge_attrs = data["edge_attrs"]
        edge_feats = data["edge_feats"]

        sender, receiver = data["edge_index"]
        num_nodes = node_feats.shape[0]

        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        del tp_weights
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        del mji
        message = self.linear(message) / self.avg_num_neighbors

        return message


class E3nnEdgeMessageBlock(torch.nn.Module):
    """This is basically MACE's RealAgnosticResidualInteractionBlock, but only up to the part
    where it computes the partial mji messages.

    It computes a "message" for each edge in the graph. Note that the message
    is different for the edge (i, j) and the edge (j, i).

    This function can be used for the preprocessing step of edges. It has no effect when used
    as the preprocessing step of nodes.
    """

    def __init__(
        self,
        irreps: Dict[str, o3.Irreps],
    ) -> None:
        super().__init__()

        node_feats_irreps = irreps["node_feats_irreps"]
        edge_attrs_irreps = irreps["edge_attrs_irreps"]
        edge_feats_irreps = irreps["edge_feats_irreps"]
        target_irreps = irreps["edge_hidden_irreps"]

        # First linear
        self.linear_up = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps,
            edge_attrs_irreps,
            target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            node_feats_irreps,
            edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        assert (
            edge_feats_irreps.lmax == 0
        ), "Edge features must be a scalar array to preserve equivariance"
        input_dim = edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.SiLU(),
        )

        irreps_mid = irreps_mid.simplify()

        self.linear = o3.Linear(irreps_mid, target_irreps)

        self.irreps_out = (None, target_irreps)

    def forward(
        self,
        data: TorchBasisMatrixData,
        node_feats: torch.Tensor,
    ) -> Tuple[None, torch.Tensor]:
        sender, receiver = data["edge_index"]

        edge_attrs = data["edge_attrs"]
        edge_feats = data["edge_feats"]

        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        del tp_weights

        # The first return is the node features [n_nodes, irreps], which we don't compute
        # The second return are the edge messages [n_edges, irreps]
        return None, self.linear(mji)
