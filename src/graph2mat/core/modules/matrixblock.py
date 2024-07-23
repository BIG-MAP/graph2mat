import numpy as np
from typing import Type, Tuple, TypeVar
from types import ModuleType

from ..data.basis import PointBasis

__all__ = ["MatrixBlock"]

# This type will be
ArrayType = TypeVar("ArrayType")


class MatrixBlock:
    """Computes a fixed size matrix coming from the product of spherical harmonics.

    There are two things to note:
        - It computes a dense matrix.
        - It computes a fixed size matrix.

    It takes care of:
      - Determining what are the irreps needed to reproduce a certain block.
      - Converting from those irreps to the actual values of the block
        using the appropiate change of basis.

    This module doesn't implement any computation, so you need to pass one
    as stated in the ``operation`` parameter.

    Parameters
    -----------
    i_irreps: o3.Irreps
        The irreps of the matrix rows.
    j_irreps: o3.Irreps
        The irreps of the matrix columns.
    symmetry: str
        Symmetries that this matrix is expected to have. This should be indicated as documented
        in `e3nn.o3.ReducedTensorProducts`. As an example, for a symmetric matrix you would
        pass "ij=ji" here.
    operation_cls: Type[torch.nn.Module]
        Torch module used to actually do the computation. On initialization, it will receive
        the `irreps_out` argument from this module, specifying the shape of the output that
        it should produce.

        On forward, this module will just be a wrapper around the operation, so you should pass
        whatever arguments that the operation expects.
    **operation_kwargs: dict
        Any arguments needed for the initialization of the `operation_cls`.

    Returns
    -----------
    matrix: ArrayType
        A 2D tensor of shape (i_irreps.dim, j_irreps.dm) containing the output matrix.
    """

    block_shape: Tuple[int, int]
    block_size: int

    symm_transpose: bool

    numpy: ModuleType = np

    def __init__(
        self,
        i_basis: PointBasis,
        j_basis: PointBasis,
        operation_cls: Type,
        symm_transpose: bool = False,
        preprocessor=None,
        **operation_kwargs,
    ):
        super().__init__()
        self.symm_transpose = symm_transpose

        self.operation = operation_cls(
            i_basis=i_basis, j_basis=j_basis, **operation_kwargs
        )

    def _compute_block(self, *args, **kwargs):
        return self.operation(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.symm_transpose == False:
            return self._compute_block(*args, **kwargs)
        else:
            forward = self._compute_block(*args, **kwargs)

            back_args = [
                (arg[1], arg[0]) if isinstance(arg, tuple) and len(arg) == 2 else arg
                for arg in args
            ]
            back_kwargs = {
                key: (value[1], value[0])
                if isinstance(value, tuple) and len(value) == 2
                else value
                for key, value in kwargs.items()
            }
            backward = self._compute_block(*back_args, **back_kwargs)

            return (forward + backward.transpose(-1, -2)) / 2

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
