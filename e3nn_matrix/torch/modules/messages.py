from e3nn import o3, nn
import torch
from typing import Union, Tuple

from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from mace.tools.scatter import scatter_sum

class EdgeMessageBlock(torch.nn.Module):
    """This is basically the RealAgnosticResidualInteractionBlock, but only up to the part
    where it computes the partial mji messages."""

    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
    ) -> None:
        super().__init__()

        # First linear
        self.linear_up = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps, edge_attrs_irreps, target_irreps,
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
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel], torch.nn.SiLU(),
        )

        irreps_mid = irreps_mid.simplify()

        self.linear = o3.Linear(irreps_mid, target_irreps)

        self.irreps_out = target_irreps

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        sender, receiver = edge_index

        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        return self.linear(mji)

class NodeMessageBlock(torch.nn.Module):
    """Basically MACE's RealAgnosticResidualInteractionBlock, without reshapes."""

    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
    ) -> None:
        super().__init__()

        # First linear
        self.linear_up = o3.Linear(
            node_feats_irreps,
            node_feats_irreps,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feats_irreps, edge_attrs_irreps, target_irreps,
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
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel], torch.nn.SiLU(),
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_mji = irreps_mid
        self.irreps_out = target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            node_feats_irreps, node_attrs_irreps, hidden_irreps
        )

        self.avg_num_neighbors = avg_num_neighbors

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors

        return message
