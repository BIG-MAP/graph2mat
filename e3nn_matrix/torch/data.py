"""Implements the Data class to use in pytorch models."""
from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Tuple, Type, Dict, Any

import sisl

from mace.tools import (
    to_one_hot,
    torch_geometric,
)

from ..data.neighborhood import get_neighborhood
from ..data.configuration import OrbitalConfiguration
from ..data.sparse import nodes_and_edges_to_sparse_orbital
from ..data.periodic_table import AtomicTableWithEdges, atomic_numbers_to_indices

class OrbitalMatrixData(torch_geometric.data.Data):
    num_nodes: torch.Tensor
    edge_index: torch.Tensor
    neigh_isc: torch.Tensor
    node_attrs: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    cell: torch.Tensor
    n_supercells: int
    nsc: torch.Tensor
    atom_labels: torch.Tensor
    edge_labels: torch.Tensor
    atom_label_ptr: torch.Tensor
    edge_label_ptr: torch.Tensor
    atom_types: torch.Tensor
    edge_types: torch.Tensor
    edge_type_nlabels: torch.Tensor
    metadata: Dict[str, Any]

    # Because we want an output in the basis of spherical harmonics, we will need to change
    # the basis. See: https://docs.e3nn.org/en/stable/guide/change_of_basis.html
    # We do the change in the inputs (coordinates, which are vectors) because it's much simpler 
    # than doing it in the outputs (spherical harmonics with arbitrary l)
    # Matrix with the change of basis to go from cartesian coordinates to spherical harmonics.
    # The minus sign is because of SIESTA's specific conventions
    _change_of_basis = torch.tensor([
        [0, 1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ], dtype=torch.get_default_dtype())
    # Save the inverse operation
    _inv_change_of_basis = torch.linalg.inv(_change_of_basis)

    def __init__(
        # All arguments must be optional in order for the get_example method of a batch to work
        self,
        edge_index: Optional[torch.Tensor]=None, # [2, n_edges]
        neigh_isc: Optional[torch.Tensor]=None, # [n_edges,]
        node_attrs: Optional[torch.Tensor]=None, # [n_nodes, n_node_feats]
        positions: Optional[torch.Tensor]=None,  # [n_nodes, 3]
        shifts: Optional[torch.Tensor]=None,  # [n_edges, 3],
        cell: Optional[torch.Tensor]=None,  # [3,3]
        nsc: Optional[torch.Tensor]=None,
        atom_labels: Optional[torch.Tensor]=None, # [total_atom_elements]
        edge_labels: Optional[torch.Tensor]=None, # [total_edge_elements]
        atom_types: Optional[torch.Tensor]=None, # [n_nodes]
        edge_types: Optional[torch.Tensor]=None, # [n_edges]
        edge_type_nlabels: Optional[torch.Tensor]=None, # [n_edge_types]
        metadata: Optional[Dict[str, Any]]=None
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0] if node_attrs is not None else None

        assert edge_index is None or (edge_index.shape[0] == 2 and len(edge_index.shape) == 2)
        assert neigh_isc is None or (neigh_isc.ndim == 1 and neigh_isc.shape[0] == edge_index.shape[1])
        assert positions is None or positions.shape == (num_nodes, 3)
        assert shifts is None or shifts.shape[1] == 3
        assert node_attrs is None or len(node_attrs.shape) == 2
        assert cell is None or cell.shape == (3, 3)
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "neigh_isc": neigh_isc,
            "n_edges": edge_index.shape[1] if edge_index is not None else None,
            "positions": positions @ self._change_of_basis.T if positions is not None else None,
            "shifts": shifts @ self._change_of_basis.T if shifts is not None else None,
            "cell": cell @ self._change_of_basis.T if cell is not None else None,
            "nsc": nsc.reshape(1, 3) if nsc is not None else None,
            "node_attrs": node_attrs,
            "atom_labels": atom_labels,
            "edge_labels": edge_labels,
            "atom_types": atom_types,
            "edge_types": edge_types,
            "edge_type_nlabels": edge_type_nlabels.reshape(1, -1) if edge_type_nlabels is not None else None,
            "metadata": metadata,
        }
        super().__init__(**data)

    @staticmethod
    def sort_edge_index(
        edge_index:np.ndarray, sc_shifts: np.ndarray, shifts: np.ndarray, edge_types: np.ndarray,
        isc_off: np.ndarray,
        inplace: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the sorted edge indices.
        
        Edges are much easier to manipulate by the block producing routines if they are ordered properly.

        This function orders edges in a way that both directions of the same edge come consecutively. 
        It also always puts first the interaction (lowest atom type, highest atom type), that is the
        one with positive edge type.

        For the unit cell, the connection in different directions is simple to understand, as it's just
        a permutation of the atoms. I.e. edges (i, j) and (j, i) are the same connection in opposite directions.
        However, for connections between supercells (if there are periodic conditions), this condition is not
        enough. The supercell shift of one direction must be the negative of the other direction. I.e. 
        only edges between (i, j, x, y, z) and (j, i, -x, -y, -z) are the same connection in opposite directions.
        It is also important to notice that in the supercell connections i and j can be the same index.

        Parameters
        -------------
        edge_index: np.ndarray of shape (2, n_edges)
            Pair of atom indices for each edge.
        sc_shifts: np.ndarray of shape (3, n_edges)
            For each edge, the number of cell boundaries the edge crosses in each
            lattice direction.
        shifts: np.ndarray of shape (3, n_edges)
            For each edge, the cartesian shift induced by sc_shifts.
        edge_types: np.ndarray of shape (n_edges, )
            For each edge, its type as an integer.
        isc_off: np.ndarrray of shape (nsc_x, nsc_y, nsc_z)
            Array that maps from sc_shifts to a single supercell index.
        inplace: bool, optional
            Whether the output should be placed in the input arrays, otherwise
            new arrays are created.

        Return
        ---------
        edge_index, sc_shifts, shifts, edge_types:
            numpy arrays with the same shape as the inputs. If inplace=True, these are
            just the input arrays, now containing the outputs.
        """
        # Get the supercell index of the neighbor in each interaction
        isc = isc_off[sc_shifts[0], sc_shifts[1], sc_shifts[2]]
        
        # Find unique edges:
        #  - For edges that are between different supercells: We get just connections from
        #   the unit cell to half of the supercells. One can then reproduce all the connections
        #   by building the same connection on the opposite direction.
        #  - For edges inside the unit cell: The connections correspond to isc == 0, however, we
        #    still have both ij and ji connections. We solve this by only taking connections where
        #    i < j. That is, the upper triangle in the "matrix of connections". Notice that self
        #    connections in the unit cell are removed before this function, so we don't have to care
        #    about them. 
        uc_edge = isc == 0
        uc_inter = edge_index[:, uc_edge]
        uc_unique_edge = uc_inter[0] < uc_inter[1]
        sc_unique_edge = isc > isc_off.size / 2

        # Manipulate unit cell connections: Get the unique interactions and make sure the pair of
        # atoms in the connection is sorted by atom type. We can easily identify the unordered connections
        # because by convention they are assigned a negative edge type. We don't need to deal with shifts
        # here because they are all 0.
        uc_unique_inter = uc_inter[:, uc_unique_edge]
        uc_unique_edge_types = edge_types[uc_edge][uc_unique_edge]
        uc_unordered = uc_unique_edge_types < 0
        uc_unique_inter[:, uc_unordered] = uc_unique_inter[::-1, uc_unordered]

        # Manipulate supercell connections: Get the unique interactions and make sure the pair of
        # atoms in the connection is sorted by atom type, doing the same as in the unit cell case.
        # However, in this case we care about the shifts because (1) we need to select the ones corresponding
        # to the selected connections and (2) we need to flip their direction for the connections that we re-sort.
        sc_unique_inter = edge_index[:, sc_unique_edge]
        sc_unique_inter_sc_shift = sc_shifts[:, sc_unique_edge]
        sc_unique_inter_shift = shifts[:, sc_unique_edge]
        sc_unique_edge_types = edge_types[sc_unique_edge]

        # Order supercell connections.
        sc_unordered = edge_types[sc_unique_edge] < 0
        sc_unique_inter[:, sc_unordered] = sc_unique_inter[::-1, sc_unordered]
        sc_unique_inter_sc_shift[:, sc_unordered] = - sc_unique_inter_sc_shift[:, sc_unordered]
        sc_unique_inter_shift[:, sc_unordered] = - sc_unique_inter_shift[:, sc_unordered]

        # Stack both unit cell and supercell connections
        unique_interactions = np.hstack([uc_unique_inter, sc_unique_inter])
        unique_sc_shifts = np.hstack([np.zeros((3, uc_unique_inter.shape[1])), sc_unique_inter_sc_shift])
        unique_shifts = np.hstack([np.zeros((3, uc_unique_inter.shape[1])), sc_unique_inter_shift])
        unique_edge_types = abs(np.concatenate([uc_unique_edge_types, sc_unique_edge_types]))

        # Now, sort edges according to absolute edge type.
        edge_sort = np.argsort(unique_edge_types)

        unique_interactions = unique_interactions[:, edge_sort]
        unique_sc_shifts = unique_sc_shifts[:, edge_sort]
        unique_shifts = unique_shifts[:, edge_sort]
        unique_edge_types = unique_edge_types[edge_sort]

        # If the operation must be done inplace, we use the input arrays as outputs,
        # otherwise we build the output arrays here, mimicking the input ones.
        if not inplace:
            edge_index = np.empty_like(edge_index)
            shifts = np.empty_like(shifts)
            sc_shifts = np.empty_like(sc_shifts)
            edge_types = np.empty_like(edge_types)

        # Make edges that belong to the same connection (but different directions) consecutive.
        edge_index[0, ::2] = edge_index[1, 1::2] = unique_interactions[0]
        edge_index[0, 1::2] = edge_index[1, ::2] = unique_interactions[1]

        # Update edge types according to the new edge indices.
        edge_types[::2] = unique_edge_types
        edge_types[1::2] = - edge_types[::2]

        # And also shifts and supercell shifts
        shifts[:, ::2] = unique_shifts
        shifts[:, 1::2] = - unique_shifts

        sc_shifts[:, ::2] = unique_sc_shifts
        sc_shifts[:, 1::2] = - unique_sc_shifts

        return edge_index, sc_shifts, shifts, edge_types

    @classmethod
    def from_config(
        cls, config: OrbitalConfiguration, z_table: AtomicTableWithEdges, sub_atomic_matrix: bool = False,
        symmetric_matrix: bool = False,
    ) -> "OrbitalMatrixData":

        indices = atomic_numbers_to_indices(config.atomic_numbers, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
        # Search for the neighbors. We use the max radius of each atom as cutoff for looking over neighbors.
        # This means that two atoms ij are neighbors if they have some overlap between their orbitals. That is
        # if distance_ij <= maxR_i + maxR_j. We subtract 0.0001 from the radius to avoid numerical problems that
        # can cause two atoms to be considered neighbors when there is no entry in the sparse matrix.
        edge_index, sc_shifts, shifts = get_neighborhood(
            positions=config.positions, cutoff=z_table.R[indices] - 1e-4, pbc=config.pbc, cell=config.cell
        )

        # Given some supercell offset of all edges, find out what is the minimum number of supercells
        # that we need to describe all interactions.
        sc_shifts = sc_shifts.T

        # Get the number of supercells needed along each direction to account for all interactions
        if config.matrix is not None:
            # If we already have a matrix, take the nsc of the matrix, which might be higher than
            # the strictly needed for the overlap of orbitals.
            # In SIESTA for example, there are the KB projectors, which add extra nonzero elements
            # for the sparse matrices.
            # However, these nonzero elements don't have any effect on the electronic density.
            nsc = config.matrix.nsc 
        else:
            nsc = abs(sc_shifts).max(axis=1) * 2 + 1

        # Then build the supercell that encompasses all of those atoms, so that we can get the
        # array that converts from sc shifts (3D) to a single supercell index. This is isc_off.
        supercell = sisl.SuperCell(config.cell, nsc=nsc)

        # Get the edge types
        edge_types = z_table.atom_type_to_edge_type(indices[edge_index])

        # Sort the edges to make it easier for the reading routines
        cls.sort_edge_index(edge_index, sc_shifts, shifts.T, edge_types, supercell.isc_off, inplace=True)

        # Count the number of labels that this matrix should have per edge type.
        unique_edge_types, counts = np.unique(abs(edge_types), return_counts=True)
        if symmetric_matrix: 
            counts = counts / 2

        edge_type_nlabels = np.zeros(z_table.edge_type[-1, -1] + 1, dtype=np.int64)
        edge_type_nlabels[unique_edge_types] = z_table.edge_block_size[unique_edge_types] * counts

        # Then, get the supercell index of each interaction.
        neigh_isc = supercell.isc_off[sc_shifts[0], sc_shifts[1], sc_shifts[2]]

        cell = (
            torch.tensor(config.cell, dtype=torch.get_default_dtype())
            if config.cell is not None
            else None
        )

        if config.matrix is not None:
            if symmetric_matrix:
                needed_edge_blocks = edge_index[:, ::2] # only unique interactions
                needed_neigh_isc = neigh_isc[::2]
            else:
                needed_edge_blocks = edge_index
                needed_neigh_isc = neigh_isc

            atom_labels, edge_labels = config.matrix.to_flat_nodes_and_edges(
                edge_index=needed_edge_blocks, edge_sc_shifts=needed_neigh_isc,
                atom_types=indices, z_table=z_table, sub_atomic_matrix=sub_atomic_matrix,
            )

            atom_labels = torch.tensor(atom_labels)
            edge_labels = torch.tensor(edge_labels)
        else:
            # We are most likely in predict mode.
            # We have no matrix data
            atom_labels = edge_labels = None


        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            neigh_isc=torch.tensor(neigh_isc, dtype=torch.int32),
            node_attrs=one_hot,
            positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            nsc=torch.tensor(supercell.nsc, dtype=torch.int16),
            atom_labels=atom_labels,
            edge_labels=edge_labels,
            atom_types=torch.tensor(indices, dtype=torch.long),
            edge_types=torch.tensor(edge_types, dtype=torch.long),
            edge_type_nlabels=torch.tensor(edge_type_nlabels, dtype=torch.int64),
            metadata=config.metadata,
        )

    def to_sparse_orbital_matrix(
        self,
        z_table: AtomicTableWithEdges,
        matrix_cls: Type[sisl.SparseOrbital],
        symmetric_matrix: bool=False,
        add_atomic_contribution: bool=False,
    ) -> sisl.SparseOrbital:

        node_labels = self.atom_labels.numpy(force=True)
        edge_labels = self.edge_labels.numpy(force=True)

        # Types for both atoms and edges.
        atom_types = self.atom_types.numpy(force=True)
        edge_types = self.edge_types.numpy(force=True)

        # Get the values for the node blocks and the pointer to the start of each block.
        node_labels_ptr = z_table.atom_block_pointer(atom_types)

        # Add back atomic contributions to the node blocks in case they were removed
        if add_atomic_contribution:
            node_labels = node_labels + np.concatenate([z_table.atomic_DM[atom_type].ravel() for atom_type in atom_types])

        # Get the values for the edge blocks and the pointer to the start of each block.
        edge_index = self.edge_index.numpy(force=True)
        neigh_isc = self.neigh_isc.numpy(force=True)
        if symmetric_matrix:
            edge_index = edge_index[:, ::2]
            edge_types = edge_types[::2]
            neigh_isc = neigh_isc[::2]

        edge_labels_ptr = z_table.edge_block_pointer(edge_types)

        # Move to numpy the rest of the relevant information
        # Change of basis back to xyz position as expected by sisl/SIESTA
        cob = self._inv_change_of_basis.numpy(force=True)
        positions = self.positions.numpy(force=True) @ cob.T
        cell = self.cell.numpy(force=True) @ cob.T
        nsc = self.nsc.numpy(force=True).squeeze()

        geometry = sisl.Geometry(
            positions,
            atoms=[z_table.atoms[at_type] for at_type in atom_types],
            sc=cell,
        )
        geometry.set_nsc(nsc)

        # Construct the matrix.
        matrix = nodes_and_edges_to_sparse_orbital(
            node_vals=node_labels, node_ptr=node_labels_ptr,
            edge_vals=edge_labels, edge_index=edge_index,
            edge_neigh_isc=neigh_isc,
            edge_ptr=edge_labels_ptr,
            geometry=geometry, sp_class=matrix_cls, symmetrize_edges=symmetric_matrix
        )

        return matrix

