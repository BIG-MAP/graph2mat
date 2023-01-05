from __future__ import annotations

from typing import Dict, Tuple, Union

from dataclasses import dataclass
import numpy as np

from ..periodic_table import AtomicTableWithEdges

OrbitalCount = np.ndarray # [num_atoms]

@dataclass
class OrbitalMatrix:
    block_dict: Dict[Tuple[int,int,int], np.ndarray]
    nsc: np.ndarray
    orbital_count: OrbitalCount

    def to_flat_nodes_and_edges(self, 
        edge_index: np.ndarray, edge_sc_shifts: np.ndarray,
        atom_order: Union[np.ndarray, None] = None,
        z_table: Union[AtomicTableWithEdges, None] = None, atom_types: Union[np.ndarray, None] = None, 
        sub_atomic_matrix: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:

        if atom_order is None:
            order = np.arange(len(self.orbital_count))
        else:
            order = atom_order
            assert len(order) == len(self.orbital_count)

        if sub_atomic_matrix:
            assert z_table is not None and atom_types is not None
            atomic_matrices = self.get_atomic_matrices(z_table)
            blocks = [
                (self.block_dict[i,i,0] - atomic_matrices[atom_types[i]]).flatten() for i in order
            ]
        else:
            blocks = [self.block_dict[i,i,0].flatten() for i in order]

        atom_labels = np.concatenate(blocks)
        
        assert edge_index.shape[0] == 2, "edge_index is assumed to be [2, n_edges]"
        blocks = [self.block_dict[edge[0],edge[1], sc_shift].flatten() for edge, sc_shift in zip(edge_index.transpose(), edge_sc_shifts)]

        edge_labels = np.concatenate(blocks)

        return atom_labels, edge_labels

    def get_atomic_matrices(self, z_table: AtomicTableWithEdges):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement a way of retreiving atomic matrices.")