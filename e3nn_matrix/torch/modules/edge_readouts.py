from abc import ABC, abstractmethod

from e3nn import o3, nn
import torch
from typing import Tuple

#from mace.modules.irreps_tools import tp_out_irreps_with_instructions

from ._symm_product import FullyConnectedSymmTensorProduct

__all__ = [
    "EdgeBlock",
    "SimpleEdgeBlock",
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

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, symm_transpose: bool = True):
        super().__init__()

        self.symm_transpose = symm_transpose

        tp_class = FullyConnectedSymmTensorProduct if symm_transpose else o3.FullyConnectedTensorProduct

        if isinstance(irreps_in, (o3.Irreps, str)):
            irreps_in = [irreps_in]

        self.tensor_products = torch.nn.ModuleList([
            tp_class(this_irreps_in, this_irreps_in, irreps_out)
            for this_irreps_in in irreps_in
        ])

    def forward(self, **tuple_kwargs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        assert len(tuple_kwargs) == len(self.tensor_products), f"Number of input tuples ({len(tuple_kwargs)}) must match number of tensor square operations ({len(self.tensor_products)})."

        tensor_tuples = iter(tuple_kwargs.values())

        final_value = self.tensor_products[0](*next(tensor_tuples))
        for i, tensor_tuple in enumerate(tensor_tuples):
            final_value = final_value + self.tensor_products[i+1](*tensor_tuple)

        return final_value
