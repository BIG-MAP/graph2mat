from typing import Union, Type

from warnings import warn

from .orbital_matrix import OrbitalMatrix
from .density_matrix import DensityMatrix

__all__ = [
    "OrbitalMatrix", "DensityMatrix",
    "get_matrix_cls"
]

_KEY_TO_MATRIX_CLS = {
    "density_matrix": DensityMatrix,
}

def get_matrix_cls(key: Union[str, None]) -> Type[OrbitalMatrix]:
    if key is None:
        return OrbitalMatrix
    else:
        key = key.lower()
        try:
            return _KEY_TO_MATRIX_CLS[key]
        except KeyError:
            warn(f"{key} is not a known matrix type key, falling back to generic OrbitalMatrix class.")
            return OrbitalMatrix