from typing import List, Tuple, Optional, Union, Dict

import torch
from e3nn import o3, nn

from graph2mat.bindings.torch import TorchBasisMatrixData

__all__ = [
    "E3nnInteraction",
    "E3nnEdgeMessageBlock",
]


# Taken directly from the MACE repository (mace.modules.irreps_tools).
def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    """"""
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions


# Taken directly from the MACE repository (mace.tools.scatter).
def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


# Taken directly from the MACE repository (mace.tools.scatter).
def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


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
