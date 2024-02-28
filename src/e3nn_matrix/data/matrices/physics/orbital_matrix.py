from __future__ import annotations

from typing import Dict, Tuple, Union

from dataclasses import dataclass, field
import numpy as np

from ..basis_matrix import BasisMatrix
from ...table import AtomicTableWithEdges

OrbitalCount = np.ndarray  # [num_atoms]


@dataclass()
class OrbitalMatrix(BasisMatrix):
    block_dict: Dict[Tuple[int, int, int], np.ndarray]
    # Size of the auxiliary supercell. This is the number of cells required
    # in each direction to account for all the interactions of the points in
    # the unit cell. If the point distribution is not periodic, this will
    # always be [1,1,1].
    nsc: np.ndarray
    # Array containing the number of basis functions for each point
    orbital_count: OrbitalCount
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
