from e3nn import o3
import torch

from typing import Tuple

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
