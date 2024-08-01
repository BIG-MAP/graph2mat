"""Conversion between different sparse representations.

Different sparse representations of a matrix are required during the different
steps of a typical workflow using `graph2mat`.
"""
from typing import Dict, Tuple, Type, Optional

import itertools

import numpy as np
from numpy.typing import ArrayLike
import sisl
from sisl.physics.sparse import SparseOrbital
from scipy.sparse import coo_array, csr_array

from .matrices import BasisMatrix, OrbitalMatrix
from ._sparse import _csr_to_block_dict


def csr_to_block_dict(
    spmat: sisl.SparseCSR,
    atoms: sisl.Atoms,
    nsc: np.ndarray,
    geometry_atoms: Optional[sisl.Atoms] = None,
    matrix_cls: Type[BasisMatrix] = OrbitalMatrix,
) -> BasisMatrix:
    """Creates a BasisMatrix object from a SparseCSR matrix

    Parameters
    ----------
    spmat :
        The sparse matrix to convert to a block dictionary.
    atoms :
        The atoms object for the matrix, containing orbital information.
    nsc :
        The auxiliary supercell size.
    matrix_cls :
        Matrix class to initialize.
    geometry_atoms :
        The atoms object for the full geometry. This allows the matrix to contain
        atoms without any orbital. Geometry atoms should contain the matrix atoms
        first and then the orbital-less atoms.
    """
    orbitals = atoms.orbitals

    block_dict = _csr_to_block_dict(
        data=spmat.data[:, 0],
        ptr=spmat.ptr,
        cols=spmat.col,
        atom_first_orb=atoms.firsto,
        orbitals=orbitals,
        n_atoms=len(atoms.specie),
    )

    orbitals = geometry_atoms.orbitals if geometry_atoms is not None else atoms.orbitals

    if issubclass(matrix_cls, OrbitalMatrix):
        return matrix_cls(block_dict=block_dict, nsc=nsc, orbital_count=orbitals)
    else:
        return matrix_cls(block_dict=block_dict, nsc=nsc, basis_count=orbitals)


def block_dict_to_coo(
    block_dict: Dict[Tuple[int, int, int], np.ndarray],
    first_orb: np.ndarray,
    n_supercells: int = 1,
    threshold: float = 1e-8,
) -> coo_array:
    """Converts a block dictionary into a coo array.

    Conversions to any other sparse structure can be done once we've got the coo array.
    """
    data = []
    rows = []
    cols = []

    no = first_orb[-1]
    for (i_at, j_at, i_sc), block in block_dict.items():
        flat_block = block.ravel()
        mask = abs(flat_block) > threshold

        data.extend(flat_block[mask])

        i_start = first_orb[i_at]
        j_start = first_orb[j_at]
        i_end = i_start + block.shape[0]
        j_end = j_start + block.shape[1]

        block_rows, block_cols = np.mgrid[i_start:i_end, j_start:j_end].reshape(2, -1)

        block_cols += no * i_sc

        rows.extend(block_rows[mask])
        cols.extend(block_cols[mask])

    return coo_array((data, (rows, cols)), (no, no * n_supercells))


def block_dict_to_csr(
    block_dict: Dict[Tuple[int, int, int], np.ndarray],
    first_orb: np.ndarray,
    n_supercells: int = 1,
    threshold: float = 1e-8,
) -> csr_array:
    """Converts a block dictionary into a csr array.

    It just uses the conversion to coo, and then converts that to a csr array.
    """
    return block_dict_to_coo(
        block_dict=block_dict,
        first_orb=first_orb,
        n_supercells=n_supercells,
        threshold=threshold,
    ).tocsr()


def nodes_and_edges_to_coo(
    node_vals: ArrayLike,
    node_ptr: ArrayLike,
    edge_vals: ArrayLike,
    edge_index: ArrayLike,
    edge_ptr: ArrayLike,
    orbitals: ArrayLike,
    n_supercells: int = 1,
    edge_neigh_isc: Optional[ArrayLike] = None,
    threshold: float = 1e-8,
    symmetrize_edges: bool = False,
) -> coo_array:
    """Converts an orbital matrix from node and edges array to coo.

    Conversions to any other sparse structure can be done once we've got the coo array.

    Parameters
    -----------
    symmetrize_edges: bool, optional
        whether for each edge only one direction is provided. The edge block for the
        opposite direction is then created as the transpose.
    """
    data = []
    rows = []
    cols = []

    first_orb = np.cumsum([0, *orbitals])
    no = first_orb[-1]

    # FIRST, FILL WITH DATA FROM NODES
    # Make sure that we are using numpy arrays (not torch tensors)
    node_vals = np.array(node_vals)

    for i_at, start in enumerate(node_ptr[:-1]):
        end = node_ptr[i_at + 1]
        flat_block = node_vals[start:end]
        mask = abs(flat_block) > threshold

        data.extend(flat_block[mask])

        dim = orbitals[i_at]

        i_start = first_orb[i_at]
        i_end = i_start + dim

        block_rows, block_cols = np.mgrid[i_start:i_end, i_start:i_end].reshape(2, -1)

        rows.extend(block_rows[mask])
        cols.extend(block_cols[mask])

    # THEN, FILL WITH DATA FROM EDGES
    # Make sure that we are using numpy arrays (not torch tensors)
    edge_vals = np.array(edge_vals)
    edge_index = np.array(edge_index)

    if edge_neigh_isc is None:
        edge_neigh_isc = itertools.repeat(0)
    else:
        edge_neigh_isc = np.array(edge_neigh_isc)

    for i_edge, ((i_at, j_at), neigh_isc) in enumerate(
        zip(edge_index.T, edge_neigh_isc)
    ):
        start, end = edge_ptr[i_edge : i_edge + 2]
        flat_block = edge_vals[start:end]
        mask = abs(flat_block) > threshold

        data.extend(flat_block[mask])

        i_start = first_orb[i_at]
        i_end = i_start + orbitals[i_at]
        j_start = first_orb[j_at]
        j_end = j_start + orbitals[j_at]

        block_rows, block_cols = np.mgrid[i_start:i_end, j_start:j_end].reshape(2, -1)
        sc_block_cols = block_cols + no * neigh_isc

        rows.extend(block_rows[mask])
        cols.extend(sc_block_cols[mask])

        if symmetrize_edges:
            # Add also the block for the opposite interaction
            data.extend(flat_block[mask])

            # Columns and rows are easy to determine if the connection is in the unit
            # cell, as the opposite block is in the transposed location.
            opp_block_cols = block_rows[mask]
            opp_block_rows = block_cols[mask]

            if neigh_isc != 0:
                # For supercell connections we need to find out what is the the supercell
                # index of the neighbor in the opposite connection.
                opp_block_cols += no * (n_supercells - neigh_isc)

            rows.extend(opp_block_rows)
            cols.extend(opp_block_cols)

    return coo_array((data, (rows, cols)), (no, no * n_supercells))


def nodes_and_edges_to_csr(
    node_vals: ArrayLike,
    node_ptr: ArrayLike,
    edge_vals: ArrayLike,
    edge_index: ArrayLike,
    edge_ptr: ArrayLike,
    orbitals: ArrayLike,
    n_supercells: int = 1,
    edge_neigh_isc: Optional[ArrayLike] = None,
    threshold: float = 1e-8,
    symmetrize_edges: bool = False,
) -> csr_array:
    """Converts an orbital matrix from node and edges array to csr.

    It just uses the conversion to coo, and then converts that to a csr array.
    """
    return nodes_and_edges_to_coo(
        node_vals=node_vals,
        node_ptr=node_ptr,
        edge_vals=edge_vals,
        edge_index=edge_index,
        edge_neigh_isc=edge_neigh_isc,
        edge_ptr=edge_ptr,
        orbitals=orbitals,
        n_supercells=n_supercells,
        threshold=threshold,
        symmetrize_edges=symmetrize_edges,
    ).tocsr()


def csr_to_sisl_sparse_orbital(
    csr: csr_array,
    geometry: sisl.Geometry,
    sp_class: Type[SparseOrbital] = SparseOrbital,
) -> SparseOrbital:
    """Converts a scipy CSR array to a sisl sparse orbital matrix."""
    return sp_class.fromsp(geometry, csr)


def nodes_and_edges_to_sparse_orbital(
    node_vals: ArrayLike,
    node_ptr: ArrayLike,
    edge_vals: ArrayLike,
    edge_index: ArrayLike,
    edge_ptr: ArrayLike,
    geometry: sisl.Geometry,
    sp_class: Type[SparseOrbital] = SparseOrbital,
    edge_neigh_isc: Optional[ArrayLike] = None,
    threshold: float = 1e-8,
    symmetrize_edges: bool = False,
) -> SparseOrbital:
    new_csr = nodes_and_edges_to_csr(
        node_vals=node_vals,
        node_ptr=node_ptr,
        edge_vals=edge_vals,
        edge_index=edge_index,
        edge_neigh_isc=edge_neigh_isc,
        edge_ptr=edge_ptr,
        orbitals=geometry.orbitals,
        n_supercells=geometry.n_s,
        threshold=threshold,
        symmetrize_edges=symmetrize_edges,
    )

    return csr_to_sisl_sparse_orbital(new_csr, geometry=geometry, sp_class=sp_class)
