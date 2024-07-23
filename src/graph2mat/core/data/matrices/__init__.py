"""Containers to store the raw matrices as a dictionary of blocks.

The matrices are stored in this format in `BasisConfiguration`, until
they are converted to flat arrays for training in `BasisMatrixData`.

However, the user does not need to initialize these matrices explicitly,
they are initialized appropiately when initializing a `BasisConfiguration`
object using the `OrbitalConfiguration.new` method.

There are different matrix classes. This is something that is probably
not needed and is reminiscent of the initial development stages.
"""
from typing import Union, Type

from warnings import warn

import sisl

from .basis_matrix import BasisMatrix
from .physics.orbital_matrix import OrbitalMatrix
from .physics.density_matrix import DensityMatrix

__all__ = ["BasisMatrix", "OrbitalMatrix", "DensityMatrix", "get_matrix_cls"]

_KEY_TO_MATRIX_CLS = {
    "density_matrix": DensityMatrix,
    sisl.DensityMatrix: DensityMatrix,
}


def get_matrix_cls(key: Union[str, sisl.SparseOrbital, None]) -> Type[OrbitalMatrix]:
    if key is None:
        return OrbitalMatrix
    else:
        if isinstance(key, str):
            key = key.lower()
        try:
            return _KEY_TO_MATRIX_CLS[key]
        except KeyError:
            warn(
                f"{key} is not a known matrix type key, falling back to generic OrbitalMatrix class."
            )
            return OrbitalMatrix
