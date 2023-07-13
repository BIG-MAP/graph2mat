from typing import List, Sequence, Union, Generator, Literal, Optional, Callable

import itertools
from pathlib import Path
from io import StringIO

import numpy as np
import sisl

from .basis import PointBasis, BasisConvention, get_change_of_basis

class BasisTableWithEdges:
    """Stores the unique types of points in the system, with their basis and the possible edges."""

    basis: List[PointBasis]
    basis_convention: BasisConvention
    types: List[Union[str, int]]

    point_matrix: Union[List[np.ndarray], None]
    edge_type: np.ndarray
    R: np.ndarray
    basis_size: np.ndarray
    atom_block_shape: np.ndarray
    atom_block_size: np.ndarray
    edge_block_shape: np.ndarray
    edge_block_size: np.ndarray

    change_of_basis: np.ndarray
    change_of_basis_inv: np.ndarray

    # These are used for saving the object in a more
    # human readable and portable way than regular pickling.
    file_names: Optional[List[str]]
    file_contents: Optional[List[str]]

    def __init__(self, basis: Sequence[PointBasis], get_point_matrix: Optional[Callable] = None):
        self.basis = list(basis)

        self.types = [point_basis.type for point_basis in self.basis]
        assert len(set(self.types)) == len(self.basis), f"The tag of each basis must be unique. Got {self.types}."

        # Define the basis convention and make sure that all the point basis adhere to that convention.
        basis_convention = self.basis[0].basis_convention

        all_conventions = [point_basis.basis_convention for point_basis in self.basis]
        assert len(set(all_conventions)) == 1 and all_conventions[0] == basis_convention, \
            f"All point basis must have the same convention. Requested convention: {basis_convention}. Basis conventions {all_conventions}."
        
        self.basis_convention = basis_convention

        # For the basis convention, get the matrices to change from cartesian to our convention.
        self.change_of_basis, self.change_of_basis_inv = get_change_of_basis(self.basis_convention)

        n_types = len(self.types)
        # Array to get the edge type from atom types.
        point_types_to_edge_types = np.empty((n_types, n_types), dtype=np.int32)
        edge_type = 0
        for i in range(n_types):
            # The diagonal edge type, always positive
            point_types_to_edge_types[i, i] = edge_type
            edge_type += 1

            # The non diagonal edge types, which are negative for the lower triangular part,
            # to account for the fact that the direction is different.
            for j in range(i + 1, n_types):
                point_types_to_edge_types[i, j] = edge_type
                point_types_to_edge_types[j, i] = - edge_type
                edge_type += 1

        self.edge_type = point_types_to_edge_types

        # Get the point matrix for each type. This is the matrix that a point would
        # have if it was the only one in the system, and it depends only on the type.
        if get_point_matrix is None:
            self.point_matrix = None
        else:
            self.point_matrix = [
                get_point_matrix(point_basis) for point_basis in self.basis
            ]

        # Get also the cutoff radii for each atom.
        self.R = np.array([point_basis.maxR() for point_basis in self.basis])

        # Store the sizes of each atom's basis.
        self.basis_size = np.array([basis.basis_size for basis in self.basis], dtype=np.int32)

        # And also the sizes of the blocks.
        self.point_block_shape = np.array([self.basis_size, self.basis_size])
        self.point_block_size = self.basis_size ** 2

        point_types_combinations = np.array(list(itertools.combinations_with_replacement(range(n_types), 2))).T
        self.edge_block_shape = self.basis_size[point_types_combinations]
        self.edge_block_size = self.edge_block_shape.prod(axis=0)

    def __str__(self):
        return "\n".join([
        f"\t- {point_basis}" for point_basis in self.basis
        ])

    def __len__(self):
        return len(self.basis)
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        same = all(x==y for x, y in itertools.zip_longest(self.basis, other.basis))
        same &= all(x==y for x, y in itertools.zip_longest(self.types, other.types))

        if self.point_matrix is None:
            same &= other.point_matrix is None
        else:
            if other.point_matrix is None:
                return False
            same &= all(np.allclose(x, y) for x, y in itertools.zip_longest(self.point_matrix, other.point_matrix))

        same &= np.allclose(self.edge_type, other.edge_type)
        same &= np.allclose(self.R, other.R)
        same &= np.allclose(self.basis_size, other.basis_size)
        same &= np.allclose(self.point_block_shape, other.point_block_shape)
        same &= np.allclose(self.point_block_size, other.point_block_size)
        same &= np.allclose(self.edge_block_shape, other.edge_block_shape)
        same &= np.allclose(self.edge_block_size, other.edge_block_size)
        return same

    def index_to_type(self, index: int) -> Union[str, int]:
        return self.types[index]

    def type_to_index(self, point_type: Union[str, int]) -> int:
        return self.types.index(point_type)
    
    def types_to_indices(self, types: Sequence) -> np.ndarray:
        # Get the unique types and the inverse indices to reconstruct the original array
        unique_types, inverse_indices = np.unique(types, return_inverse=True)

        # Now convert from types to indices
        unique_indices = np.array([self.type_to_index(unique_type) for unique_type in unique_types])

        # And reconstruct the original array, which is now an array of indices instead of types
        return unique_indices[inverse_indices]

    def point_type_to_edge_type(self, point_type:np.ndarray) -> Union[int, np.ndarray]:
        """Converts from an array of shape (2, n_edges) containing the pair
        of point types for each edge to an array of shape (n_edges,) containing
        its edge type."""
        return self.edge_type[point_type[0], point_type[1]]

    def maxR(self) -> float:
        """Returns the maximum cutoff radius in the basis."""
        return self.R.max()

    def point_block_pointer(self, point_types: Sequence[int]):
        pointers = np.zeros(len(point_types) + 1, dtype=np.int32)
        np.cumsum(self.point_block_size[point_types], out=pointers[1:])
        return pointers

    def edge_block_pointer(self, edge_types: Sequence[int]):
        pointers = np.zeros(len(edge_types) + 1, dtype=np.int32)
        np.cumsum(self.edge_block_size[edge_types], out=pointers[1:])
        return pointers

class AtomicTableWithEdges(BasisTableWithEdges):

    atoms: List[sisl.Atom]

    # These are used for saving the object in a more
    # human readable and portable way than regular pickling.
    file_names: Optional[List[str]]
    file_contents: Optional[List[str]]

    def __init__(self, atoms: Sequence[sisl.Atom]):
        from .matrices.physics.density_matrix import get_atomic_DM

        self.atoms = list(atoms)

        basis = [PointBasis.from_sisl_atom(atom) for atom in self.atoms]

        super().__init__(basis=basis, get_point_matrix=None)

        # Get the point matrix for each type. This is the matrix that a point would
        # have if it was the only one in the system, and it depends only on the type.
        self.point_matrix = [
            get_atomic_DM(atom) for atom in self.atoms
        ]

        self.file_names = None
        self.file_contents = None
    
    @property
    def zs(self):
        return self.types

    def atom_type_to_edge_type(self, atom_type: np.ndarray):
        return self.point_type_to_edge_type(atom_type)
    
    def atom_block_pointer(self, atom_types: Sequence[int]):
        return self.point_block_pointer(atom_types)
    
    @property
    def atom_block_shape(self):
        return self.point_block_shape
    
    @property
    def atom_block_size(self):
        return self.point_block_size
    
    @property
    def atomic_DM(self):
        return self.point_matrix

    @classmethod
    def from_basis_dir(cls, basis_dir: str, basis_ext: str = "ion.xml") -> "AtomicTableWithEdges":
        basis_path = Path(basis_dir)

        return cls.from_basis_glob(basis_path.glob(f"*.{basis_ext}"))

    @classmethod
    def from_basis_glob(cls, basis_glob: Union[str, Generator]) -> "AtomicTableWithEdges":
        if isinstance(basis_glob, str):
            basis_glob = Path().glob(basis_glob)

        basis = []
        #file_names = []
        #file_contents = []
        for basis_file in sorted(basis_glob):
            # TODO: Find out what to do with binary basis files formats
            #file_names.append(basis_file.name)
            #with open(basis_file, "r") as f:
            #    file_contents.append(f.read())
            basis.append(
                sisl.get_sile(basis_file).read_basis()
            )

        obj = cls(basis)
        #obj.file_names = file_names
        #obj.file_contents = file_contents
        return obj

    def _set_state_by_atoms(self, atoms: Sequence[sisl.Atom]):
        self.__init__(atoms)

    def _set_state_by_filecontents(self, file_names: List[str], file_contents: List[str]):
        assert len(file_names) == len(file_contents)
        atom_list = []
        for fname, fcontents in zip(file_names, file_contents):
            f = StringIO(fcontents)
            sile_class = sisl.get_sile_class(fname)
            with sile_class(f) as sile:
                 atom_list.append(sile.read_basis())
        self.__init__(atom_list)
        self.file_names = file_names.copy()
        self.file_contents = file_contents.copy()

    # Create pickling routines
    def __getstate__(self):
        """ Return the state of this object """
        if self.file_names is not None and self.file_contents is not None:
            return {"file_names": self.file_names,
                    "file_contents": self.file_contents}
        else:
            return {"atoms": self.atoms}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        file_names = d.get("file_names")
        file_contents = d.get("file_contents")
        if file_names is not None and file_contents is not None:
            self._set_state_by_filecontents(file_names, file_contents)
        else:
            self._set_state_by_atoms(d["atoms"])


