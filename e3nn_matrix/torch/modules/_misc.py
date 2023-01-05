from typing import Union, Tuple

import torch

from e3nn import o3

class PairmixBlock(torch.nn.Module):
    """Pairmix layer"""

    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
        edge_feats_irreps: o3.Irreps
    ) -> None:
        super().__init__()

        # Tensor product between the node features of the nodes that each edge connects
        self.tensor_product = o3.FullyConnectedTensorProduct(
            node_feats_irreps,
            node_feats_irreps,
            irreps_out,
            shared_weights=False,
            internal_weights=False,
        )

        # The weights of the tensor product depend on the radial function, so we get
        # them separately.
        input_dim = edge_feats_irreps.num_irreps
        self.tensor_product_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.tensor_product.weight_numel], torch.nn.SiLU(),
        )

        self.irreps_out = irreps_out

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor, # Radial functions
        edge_index: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        sender, receiver = edge_index

        tp_weights = self.tensor_product_weights(edge_feats)

        return self.tensor_product(node_feats[receiver], node_feats[sender], tp_weights)

class NonLinearTSQ(torch.nn.Module):

    def __init__(self, irreps_in, irreps_out):
        super().__init__()

        num_irreps = irreps_in.num_irreps

        scalar_irreps = o3.Irreps(str(irreps_in[0]))
        n_scalars = scalar_irreps.dim

        non_scalar_irreps = str(irreps_in[1:])
        n_non_scalars = num_irreps - n_scalars

        self.gate = nn.Gate(scalar_irreps, [torch.tanh], f"{n_non_scalars}x0e", [torch.tanh], non_scalar_irreps)
        self.gates = torch.nn.Parameter(torch.randn(n_non_scalars, dtype=torch.get_default_dtype()))

        self.tsq = o3.TensorSquare(irreps_in, irreps_out)

    def forward(self, node_feats):

        num_features = node_feats.shape[0]

        n_scalars = self.gate.act_scalars.irreps_in.dim

        inp = torch.concat(
            (node_feats[:, :n_scalars], self.gates.tile(num_features).reshape(num_features, -1), node_feats[:, n_scalars:]),
            dim=-1
        )

        return self.tsq(self.gate(inp))