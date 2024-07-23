from __future__ import annotations

from typing import Dict, Tuple, Union

from dataclasses import dataclass, field
import numpy as np

from ..basis_matrix import BasisMatrix
from ...table import AtomicTableWithEdges

OrbitalCount = np.ndarray  # [num_atoms]


@dataclass
class OrbitalMatrix(BasisMatrix):
    """Container to store the raw matrices as a dictionary of blocks.

    This class just adds some extra aliases to the `BasisMatrix` class,
    to use orbital terminology.
    """

    #: Dictionary containing the blocks of the matrix. The keys are tuples
    #: `(i, j_uc, sc_j)` where `ì` and `j_uc` are the atomic indices of the block
    #: (in the unit cell) and `sc_j` is the index of the neighboring cell where
    # `j` is located, e.g. 0 for a unit cell connection.
    block_dict: Dict[Tuple[int, int, int], np.ndarray]

    #: Size of the auxiliary supercell. This is the number of cells required
    #: in each direction to account for all the interactions of the points in
    #: the unit cell. If the point distribution is not periodic, this will
    #: always be [1,1,1].
    nsc: np.ndarray

    #: Alias for `basis_count`. Array containing the number of basis functions for each point
    orbital_count: OrbitalCount

    #: Array containing the number of basis functions for each point
    basis_count: OrbitalCount = field(init=False)

    def __post_init__(self):
        self.basis_count = self.orbital_count

    def get_point_matrices(
        self, basis_table: AtomicTableWithEdges
    ) -> Dict[int, ndarray]:
        return self.get_atomic_matrices(basis_table)

    def get_atomic_matrices(self, z_table: AtomicTableWithEdges):
        """"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a way of retreiving atomic matrices."
        )
