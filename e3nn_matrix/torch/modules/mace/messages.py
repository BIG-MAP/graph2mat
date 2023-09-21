from e3nn import o3, nn
import torch
from typing import Union, Tuple

from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from mace.tools.scatter import scatter_sum

__all__ = [
    "MACEEdgeMessageBlock",
    "MACENodeMessageBlock",
]

class MACEEdgeMessageBlock(torch.nn.Module):
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
        assert edge_feats_irreps.lmax == 0, "Edge features must be a scalar array to preserve equivariance"
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

        del tp_weights

        return self.linear(mji)

class MACENodeMessageBlock(torch.nn.Module):
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

        self.avg_num_neighbors = avg_num_neighbors

    def _load_from_state_dict(self, local_state_dict, prefix, *args, **kwargs):
        """We do a little hack here because previously there was a skip_tp module
        that did nothing. So if the training was done with previous versions of
        the code, the state dict will have a skip_tp key with a state.
        
        This function might be removed in the future when all models have been
        trained with the new code.
        """
        
        avoid_prefix = prefix + "skip_tp"
        for k in list(local_state_dict):
            if k.startswith(avoid_prefix):
                del local_state_dict[k]

        super()._load_from_state_dict(local_state_dict, prefix, *args, **kwargs)

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
