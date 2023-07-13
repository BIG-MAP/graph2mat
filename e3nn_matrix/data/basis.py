from typing import Union, Literal, Optional, Tuple

import dataclasses

import numpy as np
import sisl

from e3nn import o3

_change_of_basis_conventions = {
    "cartesian": np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float),
    "spherical": np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=float),
    "siesta_spherical": np.array([
        [0, 1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ], dtype=float),
}

for k, matrix in _change_of_basis_conventions.items():
    _change_of_basis_conventions[k] = (matrix, np.linalg.inv(matrix))

BasisConvention = Literal["cartesian", "spherical", "siesta_spherical"]

def get_change_of_basis(convention: BasisConvention) -> Tuple[np.ndarray, np.ndarray]:
    return _change_of_basis_conventions[convention]

@dataclasses.dataclass(frozen=True)
class PointBasis:

    type: Union[str, int]
    basis_convention: BasisConvention
    irreps: o3.Irreps
    R: Union[float, np.ndarray]
    radial_funcs: Optional[list] = None # Not in use for now.

    def __post_init__(self):
        assert isinstance(self.R, float) or (isinstance(self.R, np.ndarray) and len(self.R) == len(self))

        if self.radial_funcs is not None:
            assert len(self.radial_funcs) == len(self.irreps)

    def copy(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def __len__(self) -> int:
        return self.basis_size
    
    def __eq__(self, other) -> bool:
        return str(self.irreps) == str(other.irreps)
    
    def __str__(self):
        return f"Type: {self.type}. Irreps: {self.irreps}. MaxR: {self.maxR():.3f}."

    @property
    def basis_size(self) -> int:
        """Returns the number of basis functions per point."""
        return self.irreps.dim

    def maxR(self) -> float:
        """Returns the maximum reach of the basis."""
        return np.max(self.R)

    @classmethod
    def from_sisl_atom(cls, atom: "sisl.Atom", basis_convention: BasisConvention = "siesta_spherical"):
        """Creates a point basis from a sisl atom."""
        from .irreps_tools import get_atom_irreps

        return cls(
            type=atom.Z,
            basis_convention=basis_convention,
            irreps=get_atom_irreps(atom),
            R=atom.R,
        )

class SystemBasis:
    pass
