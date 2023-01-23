from typing import List, Sequence, Union, Generator, Iterable

import itertools
from pathlib import Path

import numpy as np
import sisl

class AtomicNumberTable: # From MACE
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)

class AtomicTableWithEdges(AtomicNumberTable):

    atoms: List[sisl.Atom]
    zs: List[int]

    atomic_DM: List[np.ndarray]
    edge_type: np.ndarray
    R: np.ndarray
    basis_size: np.ndarray
    atom_block_shape: np.ndarray
    atom_block_size: np.ndarray
    edge_block_shape: np.ndarray
    edge_block_size: np.ndarray

    def __init__(self, atoms: Sequence[sisl.Atom]):
        from .matrices.density_matrix import get_atomic_DM
        self.atoms = list(atoms)

        self.zs = [atom.Z for atom in atoms]

        natoms = len(atoms)
        # Array to get the edge type from atom types.
        atom_types_to_edge_type = np.empty((natoms, natoms), dtype=np.int32)
        edge_type = 0
        for i in range(natoms):
            # The diagonal edge type, always positive
            atom_types_to_edge_type[i, i] = edge_type
            edge_type += 1

            # The non diagonal edge types, which are negative for the lower triangular part,
            # to account for the fact that the direction is different.
            for j in range(i + 1, natoms):
                atom_types_to_edge_type[i, j] = edge_type
                atom_types_to_edge_type[j, i] = - edge_type
                edge_type += 1

        self.edge_type = atom_types_to_edge_type

        # Get the atomic density matrix for each atom
        self.atomic_DM = [
            get_atomic_DM(atom) for atom in atoms
        ]

        # Get also the cutoff radii for each atom.
        self.R = np.array([atom.maxR() for atom in atoms])

        # Store the sizes of each atom's basis.
        self.basis_size = np.array([atom.no for atom in atoms], dtype=np.int32)

        # And also the sizes of the blocks.
        self.atom_block_shape = np.array([self.basis_size, self.basis_size])
        self.atom_block_size = self.basis_size ** 2

        atom_combinations = np.array(list(itertools.combinations_with_replacement(range(natoms), 2))).T
        self.edge_block_shape = self.basis_size[atom_combinations]
        self.edge_block_size = self.edge_block_shape.prod(axis=0)

    def __eq__(self, other):
        same = all(x==y for x, y in itertools.zip_longest(self.atoms, other.atoms))
        same &= all(x==y for x, y in itertools.zip_longest(self.zs, other.zs))
        same &= all(np.allclose(x, y) for x, y in itertools.zip_longest(self.atomic_DM, other.atomic_DM))
        same &= np.allclose(self.edge_type, other.edge_type)
        same &= np.allclose(self.R, other.R)
        same &= np.allclose(self.basis_size, other.basis_size)
        same &= np.allclose(self.atom_block_shape, other.atom_block_shape)
        same &= np.allclose(self.atom_block_size, other.atom_block_size)
        same &= np.allclose(self.edge_block_shape, other.edge_block_shape)
        same &= np.allclose(self.edge_block_size, other.edge_block_size)
        return same

    def atom_type_to_edge_type(self, atom_type:np.ndarray) -> Union[int, np.ndarray]:
        """Converts from an array of shape (2, n_edges) containing the pair
        of atom types for each edge to an array of shape (n_edges,) containing
        the edge type for each edge"""
        return self.edge_type[atom_type[0], atom_type[1]]

    def maxR(self) -> float:
        """Returns the maximum cutoff radius in the basis."""
        return self.R.max()

    def atom_block_pointer(self, atom_types: Sequence[int]):
        pointers = np.zeros(len(atom_types) + 1, dtype=np.int32)
        np.cumsum(self.atom_block_size[atom_types], out=pointers[1:])
        return pointers

    def edge_block_pointer(self, edge_types: Sequence[int]):
        pointers = np.zeros(len(edge_types) + 1, dtype=np.int32)
        np.cumsum(self.edge_block_size[edge_types], out=pointers[1:])
        return pointers

    @classmethod
    def from_basis_dir(cls, basis_dir: str, basis_ext: str = "ion.xml") -> "AtomicTableWithEdges":
        basis_path = Path(basis_dir)

        return cls.from_basis_glob(basis_path.glob(f"*.{basis_ext}"))

    @classmethod
    def from_basis_glob(cls, basis_glob: Union[str, Generator]) -> "AtomicTableWithEdges":
        if isinstance(basis_glob, str):
            basis_glob = Path().glob(basis_glob)

        basis = []
        for basis_file in sorted(basis_glob):
            basis.append(
                sisl.get_sile(basis_file).read_basis()
            )

        return cls(basis)

def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)
