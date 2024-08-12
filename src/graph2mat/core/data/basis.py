"""Utilities to describe a basis set for a point type."""

from typing import Union, Literal, Tuple, Sequence
from numbers import Number

import dataclasses

import numpy as np
import sisl

_change_of_basis_conventions = {
    "cartesian": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    "spherical": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
    "siesta_spherical": np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float),
}

for k, matrix in _change_of_basis_conventions.items():
    _change_of_basis_conventions[k] = (matrix, np.linalg.inv(matrix))

BasisConvention = Literal["cartesian", "spherical", "siesta_spherical"]


def get_atom_basis(atom: sisl.Atom):
    """For a given atom, returns the representation of its basis.

    Parameters
    ----------
    atom: sisl.Atom
        The atom for which we want the irreps of its basis.

    Returns
    ----------
    Tuple[int, int, int]:
        the basis representation.
    """

    if atom.no == 0:
        return []

    atom_irreps = []

    # Array that stores the number of orbitals for each l.
    # We allocate 8 ls, we will probably never need this much.
    n_ls = np.zeros(8)

    # Loop over all orbitals that this atom contains
    for orbital in atom.orbitals:
        # For each orbital, find its l quantum number
        # and increment the total number of orbitals for that l
        n_ls[orbital.l] += 1

    # We don't really want to know the number of orbitals for a given l,
    # but the number of SETS of orbitals. E.g. a set of l=1 has 3 orbitals.
    n_ls /= 2 * np.arange(8) + 1

    # Now just loop over all ls, and intialize as much irreps as we need
    # for each of them. We build a list of tuples (n_irreps, (l, parity))
    # to pass it to o3.Irreps.
    for l, n_l in enumerate(n_ls):
        if n_l != 0:
            atom_irreps.append((int(n_l), l, (-1) ** l))

    return atom_irreps


def get_change_of_basis(
    original: BasisConvention, target: BasisConvention
) -> Tuple[np.ndarray, np.ndarray]:
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
    if original == target:
        return np.eye(3), np.eye(3)

    elif original == "cartesian":
        return _change_of_basis_conventions[target]
    else:
        orig_from_cartesian, orig_to_cartesian = _change_of_basis_conventions[original]
        target_from_cartesian, target_to_cartesian = _change_of_basis_conventions[
            target
        ]
        return (orig_to_cartesian.T @ target_from_cartesian.T).T, (
            target_to_cartesian.T @ orig_from_cartesian.T
        ).T


@dataclasses.dataclass(frozen=True)
class PointBasis:
    """Stores the basis set for a point type.

    Parameters
    ----------
    type : Union[str, int]
        The type ID, e.g. some meaningful name or a number.
    basis_convention : BasisConvention
        The spherical harmonics convention used for the basis.
    basis:
        Specification of the basis set that the point type has.
        It can be a list of specifications, then each item in the list can
        be the number of sets of functions for a given l (determined
        by the position of the item in the list), or a tuple specifying
        (n_sets, l, parity).

        It can also be a string representing the irreps of the basis in
        the `e3nn` format. E.g. "3x0e+2x1o" would mean 3 `l=0` and 2 `l=1`
        sets.
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
        from graph2mat import PointBasis

        # Let's create a basis with 3 l=0 functions and 2 sets of l=1 functions.
        # The convention for spherical harmonics will be the standard one.
        # We call this type of basis set "A", and functions have a reach of 5.
        basis = PointBasis("A", R=5, basis=[3, 2], basis_convention="spherical")

        # Same but with a different reach for l=0 (R=5) and l=1 functions (R=3).
        basis = PointBasis("A", R=np.array([5, 5, 5, 3, 3, 3, 3, 3, 3]), irreps=[3, 2], basis_convention="spherical" )

        # Equivalent specification of the basis using tuples:
        basis = PointBasis("A", R=5, basis=[(3, 0, 1), (2, 1, -1)], basis_convention="spherical")

    """

    type: Union[str, int]
    R: Union[float, np.ndarray]
    basis: Union[str, Sequence[Union[int, Tuple[int, int, int]]]] = ()
    basis_convention: BasisConvention = "spherical"

    def __post_init__(self):
        basis = self._sanitize_basis(self.basis)

        object.__setattr__(self, "basis", basis)

        assert isinstance(self.R, Number) or (
            isinstance(self.R, np.ndarray) and len(self.R) == self.basis_size
        ), f"R must be a float or an array of length {self.basis_size} (the number of functions)."

    def _sanitize_basis(
        self, basis: Union[Sequence[int], str]
    ) -> Tuple[Tuple[int, int, int], ...]:
        """Sanitizes the basis ensuring that it is a tuple of tuples."""

        def _get_sphericalharm_parity(l: int) -> int:
            return (-1) ** l

        def _san_basis_spec(i, basis_spec) -> Tuple[int, int, int]:
            if isinstance(basis_spec, int):
                return (basis_spec, i, _get_sphericalharm_parity(i))
            else:
                return basis_spec

        if isinstance(basis, str):
            san_basis = []

            for irreps in basis.split("+"):
                irreps = irreps.strip()

                vals = irreps.split("x")
                if len(vals) == 2:
                    mul, l = vals
                else:
                    mul = 1
                    l = vals[0]

                l = int(l.replace("e", "").replace("o", ""))
                mul = int(mul)
                san_basis.append((mul, l, _get_sphericalharm_parity(l)))

            return tuple(san_basis)
        else:
            return tuple(
                _san_basis_spec(i, basis_spec) for i, basis_spec in enumerate(basis)
            )

    @property
    def e3nn_irreps(self):
        """Returns the irreps in the e3nn format."""
        from e3nn import o3

        return o3.Irreps((mul, (l, p)) for mul, l, p in self.basis)

    def copy(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def __len__(self) -> int:
        return self.basis_size

    def __eq__(self, other) -> bool:
        return self.basis == other.basis

    def __str__(self):
        return f"Type: {self.type}. Basis: {self.basis}. MaxR: {self.maxR():.3f}."

    @property
    def basis_size(self) -> int:
        """Returns the number of basis functions per point."""
        return sum(n * (2 * l + 1) for n, l, _ in self.basis)

    @property
    def num_sets(self) -> int:
        """Returns the number of sets of functions.

        E.g. for a basis with 3 l=0 functions and 2 sets of l=1 functions, this
        returns 5.
        """
        return sum(n for n, _, _ in self.basis)

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
        return cls(
            type=atom.Z,
            basis_convention=basis_convention,
            basis=get_atom_basis(atom),
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
        for n_shells, l, parity in self.basis:
            for izeta in range(n_shells):
                for m in range(-l, l + 1):
                    orb = sisl.AtomicOrbital(n=3, l=l, m=m, zeta=izeta, R=R[i])
                    orbitals.append(orb)
                    i += 1

        return sisl.Atom(Z=Z, orbitals=orbitals)


class NoBasisAtom(sisl.Atom):
    """Placeholder for atoms without orbitals.

    This should no longer be needed once sisl allows atoms with 0 orbitals.

    Atoms with no basis are for example needed for the fitting of QM/MM
    simulations.
    """

    @property
    def no(self):
        """The number of orbitals belonging to this atom."""
        return 0

    @property
    def q0(self):
        """The initial charge of the orbitals of this atom."""
        return np.array([])
