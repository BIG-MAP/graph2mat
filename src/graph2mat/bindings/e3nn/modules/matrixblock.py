from e3nn import o3
import torch
from typing import Type, Dict
import inspect

from graph2mat import PointBasis
from graph2mat.bindings.torch import TorchMatrixBlock

__all__ = ["E3nnIrrepsMatrixBlock"]


class E3nnIrrepsMatrixBlock(TorchMatrixBlock):
    """Computes a matrix block by computing its irreps first."""

    def __init__(
        self,
        i_basis: PointBasis,
        j_basis: PointBasis,
        symmetry: str,
        operation_cls: Type,
        symm_transpose: bool = False,
        preprocessor=None,
        irreps: Dict[str, o3.Irreps] = {},
        **operation_kwargs,
    ):
        torch.nn.Module.__init__(self)

        i_irreps = i_basis.e3nn_irreps
        j_irreps = j_basis.e3nn_irreps

        self.i_irreps = i_irreps
        self.j_irreps = j_irreps

        self.setup_reduced_tp(i_irreps=i_irreps, j_irreps=j_irreps, symmetry=symmetry)
        self.symm_transpose = symm_transpose

        operation_kwargs = {
            **self.get_init_kwargs(irreps, operation_cls),
            **operation_kwargs,
        }

        self.operation = operation_cls(**operation_kwargs)

    def get_summary(self):
        return f"{str(self.operation.__class__.__name__)}: ({self.i_irreps}) x ({self.j_irreps}) -> {self._irreps_out}"

    def setup_reduced_tp(self, i_irreps: o3.Irreps, j_irreps: o3.Irreps, symmetry: str):
        # Store the shape of the block.
        self.block_shape = (i_irreps.dim, j_irreps.dim)
        # And number of elements in the block.
        self.block_size = i_irreps.dim * j_irreps.dim

        # Understand the irreps out that we need in order to create the block.
        # The block is a i_irreps.dim X j_irreps.dim matrix, with possible symmetries that can
        # reduce the number of degrees of freedom. We indicate this to the ReducedTensorProducts,
        # which we only use as a helper.
        reduced_tp = o3.ReducedTensorProducts(symmetry, i=i_irreps, j=j_irreps)
        self._irreps_out = reduced_tp.irreps_out

        # We also store the change of basis, a matrix that will bring us from the irreps_out
        # to the actual matrix block that we want to calculate.
        self.register_buffer("change_of_basis", reduced_tp.change_of_basis)

    def _compute_block(self, *args, **kwargs):
        # Get the irreducible output
        irreducible_out = self.operation(*args, **kwargs)

        # And convert it to the actual block of the matrix, using the change of basis
        # matrix stored on initialization.
        # n = number of nodes, i = dim of irreps, x = rows in block, y = cols in block
        return self.numpy.einsum("ni,ixy->nxy", irreducible_out, self.change_of_basis)

    def get_init_kwargs(self, irreps: Dict[str, o3.Irreps], operation_cls) -> dict:
        kwargs = {}
        op_sig = inspect.signature(operation_cls)

        irreps = {**irreps}

        irreps["irreps_in"] = [
            irrep
            for irrep in [
                irreps["node_feats_irreps"],
                irreps.get("edge_messages_irreps"),
            ]
            if irrep is not None
        ]
        irreps["irreps_out"] = self._irreps_out

        for k in op_sig.parameters:
            if k in irreps:
                kwargs[k] = irreps[k]

        return kwargs
