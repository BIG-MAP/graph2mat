"""Utilities for managing basis sets."""

from typing import Union, Literal, Tuple
from numbers import Number

import dataclasses

import numpy as np
import sisl

from e3nn import o3

_change_of_basis_conventions = {
    "cartesian": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    "spherical": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
    "siesta_spherical": np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float),
}

for k, matrix in _change_of_basis_conventions.items():
    _change_of_basis_conventions[k] = (matrix, np.linalg.inv(matrix))

BasisConvention = Literal["cartesian", "spherical", "siesta_spherical"]


def get_change_of_basis(convention: BasisConvention) -> Tuple[np.ndarray, np.ndarray]:
    """Change of basis matrix for the given convention.

    Parameters
    ----------
    convention:
        The convention for spherical harmonics.

    Returns
    ----------
    change_of_basis_matrix
    inverse_change_of_basis
    """
    return _change_of_basis_conventions[convention]


@dataclasses.dataclass(frozen=True)
class PointBasis:
    """Stores the basis set for a point type.

    Parameters
    ----------
    type : Union[str, int]
        The type ID, e.g. some meaningful name or a number.
    basis_convention : BasisConvention
        The spherical harmonics convention used for the basis.
    irreps : o3.Irreps
        Irreps of the basis. E.g. ``o3.Irreps("3x0e + 2x1o")``
        for a basis with 3 l=0 functions and 2 sets of l=1 functions.

        ``o3.Irreps("")``, the default value, means that this point
        has no basis functions.
    R : Union[float, np.ndarray]
        The reach of the basis.
        If a float, the same reach is used for all functions.

        If an array, the reach is different for each SET of functions. E.g.
        for a basis with 3 l=0 functions and 2 sets of l=1 functions, you must
        provide an array of length 5.

        The reach of the functions will determine if the point interacts with
        other points.

    Examples
    ----------

    .. code-block:: python

        import numpy as np

        from e3nn import o3
        from e3nn_matrix.data import PointBasis

        # Let's create a basis with 3 l=0 functions and 2 sets of l=1 functions.
        # The convention for spherical harmonics will be the standard one.
        # We call this type of basis set "A", and functions have a reach of 5.
        basis = PointBasis("A", R=5, irreps=o3.Irreps("3x0e + 2x1o"), basis_convention="spherical")

        # Same but with a different reach for l=0 (R=5) and l=1 functions (R=3).
        basis = PointBasis("A", R=np.array([5, 5, 5, 3, 3, 3, 3, 3, 3]), irreps=o3.Irreps("3x0e + 2x1o"), basis_convention="spherical" )

    """

    type: Union[str, int]
    R: Union[float, np.ndarray]
    irreps: o3.Irreps = o3.Irreps("")
    basis_convention: BasisConvention = "spherical"

    def __post_init__(self):
        assert isinstance(self.R, Number) or (
            isinstance(self.R, np.ndarray) and len(self.R) == self.basis_size
        ), f"R must be a float or an array of length {self.basis_size} (the number of functions)."

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

    @property
    def num_sets(self) -> int:
        """Returns the number of sets of functions.

        E.g. for a basis with 3 l=0 functions and 2 sets of l=1 functions, this
        returns 5.
        """
        return self.irreps.num_irreps

    def maxR(self) -> float:
        """Returns the maximum reach of the basis."""
        return np.max(self.R)

    @classmethod
    def from_sisl_atom(
        cls, atom: "sisl.Atom", basis_convention: BasisConvention = "siesta_spherical"
    ):
        """Creates a point basis from a sisl atom.

        Parameters
        ----------
        atom:
            The atom from which to create the basis.
        basis_convention:
            The spherical harmonics convention used for the basis.
        """
        from .irreps_tools import get_atom_irreps

        return cls(
            type=atom.Z,
            basis_convention=basis_convention,
            irreps=get_atom_irreps(atom),
            R=atom.R if atom.no != 0 else atom.R[0],
        )

    def to_sisl_atom(self, Z: int = 1) -> "sisl.Atom":
        """Converts the basis to a sisl atom.

        Parameters
        ----------
        Z:
            The atomic number of the atom.
        """

        import sisl

        if self.basis_size == 0:
            return NoBasisAtom(Z=Z, R=self.R)

        orbitals = []

        R = (
            self.R
            if isinstance(self.R, np.ndarray)
            else np.full((self.basis_size,), self.R)
        )

        i = 0
        for x in self.irreps:
            l = x.ir.l
            n_shells = x.mul
            for izeta in range(n_shells):
                for m in range(-l, l + 1):
                    orb = sisl.AtomicOrbital(n=3, l=l, m=m, zeta=izeta, R=R[i])
                    orbitals.append(orb)
                    i += 1

        return sisl.Atom(Z=Z, orbitals=orbitals)

class NoBasisAtom(sisl.Atom):
    """Placeholder for atoms without orbitals.
    
    This should no longer be needed once sisl allows atoms with 0 orbitals."""

    @property
    def no(self):
        return 0
    
    @property
    def q0(self):
        return np.array([])
    