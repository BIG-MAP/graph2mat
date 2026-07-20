"""Conversion between different sparse representations.

Different sparse representations of a matrix are required during the different
steps of a typical workflow using `graph2mat`.
"""
from typing import Dict, Tuple, Type, Optional, Callable, Any

import itertools
from functools import partial

import numpy as np
from numpy.typing import ArrayLike
import sisl
from sisl.physics.sparse import SparseOrbital
from scipy.sparse import coo_array, csr_array

from .matrices import BasisMatrix, OrbitalMatrix
from .formats import Formats, conversions
from ._sparse import _csr_to_block_dict


def register_all_sisl(source, target, converter, manager):
    if target != Formats.SISL:
        return

    specific_targets = [
        (Formats.SISL_H, sisl.Hamiltonian),
        (Formats.SISL_DM, sisl.DensityMatrix),
        (Formats.SISL_EDM, sisl.EnergyDensityMatrix),
    ]

    for specific_target, sp_class in specific_targets:
        if manager.has_converter(source, specific_target):
            continue

        specific_converter = partial(converter, sp_class=sp_class)
        manager.register_converter(source, specific_target, specific_converter)


conversions.add_callback(register_all_sisl)

# -----------------------------------------------
#            Conversion functions
# -----------------------------------------------

converter = conversions.converter


@converter
def _csr_to_numpy(csr: csr_array) -> np.ndarray:
    return csr.toarray()


@converter
def _coo_to_csr(coo: coo_array) -> csr_array:
    return coo.tocsr()


@converter
def _csr_to_coo(csr: csr_array) -> coo_array:
    return csr.tocoo()


@converter
def _coo_to_numpy(coo: coo_array) -> np.ndarray:
    return coo.toarray()


@converter(Formats.SCIPY_CSR, Formats.BASISMATRIX)
def csr_to_block_dict(
    spmat: sisl.SparseCSR,
    atoms: sisl.Atoms,
    nsc: np.ndarray,
    geometry_atoms: Optional[sisl.Atoms] = None,
    matrix_cls: Type[BasisMatrix] = OrbitalMatrix,
    fill_value: float = 0,
) -> BasisMatrix:
    """Creates a BasisMatrix object from a SparseCSR matrix

    Parameters
    ----------
    spmat
        The sparse matrix to convert to a block dictionary.
    atoms
        The atoms object for the matrix, containing orbital information.
    nsc
        The auxiliary supercell size.
    matrix_cls
        Matrix class to initialize.
    geometry_atoms
        The atoms object for the full geometry. This allows the matrix to contain
        atoms without any orbital. Geometry atoms should contain the matrix atoms
        first and then the orbital-less atoms.
    fill_value
        The value to use for the empty elements in the matrix. Use `np.nan` if the
        matrix is not really sparse (empty elements are just elements that you don't
        want to fit) like the density matrix. Models will not attempt to fit the
        `np.nan` values.
    """
    orbitals = atoms.orbitals

    block_dict = _csr_to_block_dict(
        data=spmat.data[:, 0],
        ptr=spmat.ptr,
        cols=spmat.col,
        atom_first_orb=atoms.firsto,
        orbitals=orbitals,
        n_atoms=len(atoms.species),
        fill_value=fill_value,
    )

    orbitals = geometry_atoms.orbitals if geometry_atoms is not None else atoms.orbitals

    if issubclass(matrix_cls, OrbitalMatrix):
        return matrix_cls(block_dict=block_dict, nsc=nsc, orbital_count=orbitals)
    else:
        return matrix_cls(block_dict=block_dict, nsc=nsc, basis_count=orbitals)


@converter(Formats.BLOCK_DICT, Formats.SCIPY_COO)
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


def _blockmatrix_coo_coords(
    orbitals_row: ArrayLike,
    orbitals_col: ArrayLike,
    edge_index: ArrayLike,
    n_supercells: int = 1,
    edge_neigh_isc: Optional[ArrayLike] = None,
    symmetrize_edges: bool = False,
):
    """Returns the coo cordinates of a block matrix.

    This function assumes that:

        - All node blocks contain nonzero entries.
        - Edge blocks for edges included in `edge_index` contain
        nonzero entries.
        - All individual blocks are dense. I.e. if a block "exists",
        there are non-zero entries for all of its elements.

    The order of the coordinates returned by this function is:

        1. Node blocks.
        2. Edge blocks.
        3. Edge blocks in reverse direction (if `symmetrize_edges == True`).

    Blocks (1) and (2) are assumed to be in row-major order. Blocks (3), if
    any, are assumed to contain the data in the exact same order as (2).

    Parameters
    ----------
    orbitals_row / orbitals_col:
        for each atom, the amount of orbitals it has. In case of contracted basis sets,
        rows represent the original basis, and columns the contracted basis. In case of
        non-contracted basis sets, both are equal.

    edge_index:
        shape (2, n_edges), for each edge the indices of the atoms
        that participate. If `symmetrize_edges` is `True`, this must
        ONLY contain the edges in one of the directions.
        # TODO: verify this work with non sym in the non-square case.
    n_supercells:
        number of supercells in the matrix.
    edge_neigh_isc:
        shape (n_edges, ), for each edge the index of the supercell
        of the interaction. If not provided, all interactions are assumed
        to be in the unit cell.
    symmetrize_edges:
        whether we should assume that the matrix contains also the edges
        that are in the opposite direction as the ones provided in
        `edge_index`.
    """
    # Initialize the arrays to store the coordinates.
    rows = []
    cols = []

    # BORRAR
    # print(f"In sparse.py _blockmatrix_coo_coords:")
    # print(f"orbitals_row: {orbitals_row}")
    # print(f"orbitals_col: {orbitals_col}")
    # print(f"edge_index: {edge_index}")

    is_square = np.array_equal(orbitals_row, orbitals_col)


    # Store index of first orbital for each atom, as well as total number of orbitals.
    first_orb_row = np.cumsum([0, *orbitals_row])
    first_orb_col = np.cumsum([0, *orbitals_col])
    no_row = first_orb_row[-1]  # Total number of orbitals in the row.
    no_col = first_orb_col[-1]  # Total number of orbitals in the column.

    # First, compute the coordinates for the node blocks.
    for dim in range(len(orbitals_row)):
        i_start_row = first_orb_row[dim]
        i_end_row = i_start_row + orbitals_row[dim]
        j_start_col = first_orb_col[dim]
        j_end_col = j_start_col + orbitals_col[dim]

        block_rows, block_cols = np.mgrid[i_start_row:i_end_row, j_start_col:j_end_col].reshape(2, -1)

        rows.extend(block_rows)
        cols.extend(block_cols)

    # Then, the coordinates for the edge blocks.

    # Initialize lists for symmetrized edges, which we store separately
    # so that we can append all of them at the end.
    rows_symm = []
    cols_symm = []

    # Assume unit cell interactions if edge_neigh_isc is not provided
    if edge_neigh_isc is None:
        edge_neigh_isc = itertools.repeat(0)
    else:
        edge_neigh_isc = np.array(edge_neigh_isc)

    for i_edge, ((i_at, j_at), neigh_isc) in enumerate(
        zip(edge_index.T, edge_neigh_isc)
    ): 
        if neigh_isc != 0 and not is_square:
            raise NotImplementedError(
                "Supercell interactions are not yet implemented for non-square matrices (contractions)."
            )
        i_start = first_orb_row[i_at]
        i_end = i_start + orbitals_row[i_at]
        j_start = first_orb_col[j_at]
        j_end = j_start + orbitals_col[j_at]

        block_rows, block_cols = np.mgrid[i_start:i_end, j_start:j_end].reshape(2, -1)
        sc_block_cols = block_cols + no_col * neigh_isc

        rows.extend(block_rows)
        cols.extend(sc_block_cols)

        if symmetrize_edges and is_square:
            # Columns and rows are easy to determine if the connection is in the unit
            # cell, as the opposite block is in the transposed location.
            opp_block_cols = block_rows
            opp_block_rows = block_cols

            if neigh_isc != 0:
                # For supercell connections we need to find out what is the the supercell
                # index of the neighbor in the opposite connection.
                opp_block_cols += no_col * (n_supercells - neigh_isc)

            rows_symm.extend(opp_block_rows)
            cols_symm.extend(opp_block_cols)
        elif symmetrize_edges and not is_square:
            raise ValueError(
                "SN: TODO - I think thsi not work as I intended to... maybe it cannot be")
            # We assume that the edge_index contains only the edges in one direction,
            # but as it it not square, we cannot assume that the opposite direction is just the transpose of the
            # original edge.
            i_start_symm = first_orb_row[j_at]
            i_end_symm = i_start_symm + orbitals_row[j_at]
            j_start_symm = first_orb_col[i_at]
            j_end_symm = j_start_symm + orbitals_col[i_at]
            block_rows_symm, block_cols_symm = np.mgrid[i_start_symm:i_end_symm, j_start_symm:j_end_symm].reshape(2, -1)
            rows_symm.extend(block_rows_symm)
            cols_symm.extend(block_cols_symm)
    # Add coordinates of symmetrized edges to the list of coordinates.
    rows.extend(rows_symm)
    cols.extend(cols_symm)

    # BORRAR
    # print(f"Len rows in _blockmatrix_coo_coords: {len(rows)}")
    # print(f"Len cols in _blockmatrix_coo_coords: {len(cols)}")

    return np.array(rows), np.array(cols), (no_row, no_col * n_supercells)


def _nodes_and_edges_to_coo(
    concatenate: Callable[[tuple[ArrayLike, ArrayLike, ArrayLike]], ArrayLike],
    init_coo: Callable[[ArrayLike, np.ndarray, np.ndarray, tuple[int, int]], Any],
    node_vals: ArrayLike,
    edge_vals: ArrayLike,
    edge_index: ArrayLike,
    orbitals_row: ArrayLike,
    orbitals_col: ArrayLike,
    n_supercells: int = 1,
    edge_neigh_isc: Optional[ArrayLike] = None,
    threshold: Optional[float] = None,
    symmetrize_edges: bool = False,
) -> Any:
    """Converts an orbital matrix from node and edges array to coo.

    This is a generic function that can be used to convert to any coo format
    (e.g. scipy coo or torch coo).

    Parameters
    ----------
    concatenate
        function to concatenate the node and edge values to generate the full array
        of values. It receives a list of arrays like:

        - [node_vals, edge_vals, edge_vals] if ``symmetrize_edges`` is ``True``.
        - [node_vals, edge_vals] otherwise.
    init_coo
        function to initialize the coo array. It receives four arguments:

        - The matrix values (as returned by ``concatenate``).
        - The rows corresponding to the matrix values.
        - The columns corresponding to the matrix values.
        - The shape of the matrix.
    node_vals
        Flat array containing the values of the node blocks.
        The order of the values is first by node index, then row then column.
    edge_vals
        Flat array containing the values of the edge blocks.
        The order of the values is first by edge index, then row then column.
    edge_index
        Array of shape (2, n_edges) containing the indices of the atoms
        that participate in each edge.
    orbitals_row / orbitals_col
        Array of shape (n_nodes, ) containing the number of orbitals for each atom.
        For contracted basis sets, rows represent the original basis, and columns the
        contracted basis. For non-contracted basis sets, both are equal.
    n_supercells
        Number of auxiliary supercells.
    edge_neigh_isc
        Array of shape (n_edges, ) containing the supercell index of the second atom
        in each edge with respect to the first atom.
        If not provided, all interactions are assumed to be in the unit cell.
    threshold
        Matrix elements with a value below this number are set to 0.
    symmetrize_edges
        whether for each edge only one direction is provided. The edge block for the
        opposite direction is then created as the transpose.
    """

    # BORRAR
    # print(f"In sparse.py _nodes_and_edges_to_coo:")
    # print(f"calling _blockmatrix_coo_coords with:")
    # print(f"orbitals_row: {orbitals_row}")
    # print(f"orbitals_col: {orbitals_col}")
    # print(f"edge_index: {edge_index}")
    # print(f"n_supercells: {n_supercells}")
    # print(f"edge_neigh_isc: {edge_neigh_isc}")
    # print(f"symmetrize_edges: {symmetrize_edges}")

    rows, cols, shape = _blockmatrix_coo_coords(
        orbitals_row=orbitals_row,
        orbitals_col=orbitals_col,
        edge_index=edge_index,
        n_supercells=n_supercells,
        edge_neigh_isc=edge_neigh_isc,
        symmetrize_edges=symmetrize_edges,
    )
    # BORRAR
    # print(f"Len rows in _nodes_and_edges_to_coo: {len(rows)}")
    # print(f"Len cols in _nodes_and_edges_to_coo: {len(cols)}")
    # print(f"Shape in _nodes_and_edges_to_coo: {shape}")
    # print(f"Len node_vals in _nodes_and_edges_to_coo: {len(node_vals)}")
    # print(f"Len edge_vals in _nodes_and_edges_to_coo: {len(edge_vals)}")

    if symmetrize_edges:
        sparse_data = concatenate([node_vals, edge_vals, edge_vals])
    else:
        sparse_data = concatenate([node_vals, edge_vals])
    # BORRAR
    # print(f"Len sparse_data in _nodes_and_edges_to_coo: {len(sparse_data)}")

    if threshold is not None:
        mask = abs(sparse_data) > threshold
    else:
        # Remove NaNs
        mask = sparse_data == sparse_data

    sparse_data = sparse_data[mask]
    rows = rows[mask]
    cols = cols[mask]

    return init_coo(sparse_data, rows, cols, shape)


@converter(Formats.NODESEDGES, Formats.SCIPY_COO)
def nodes_and_edges_to_coo(
    node_vals: np.ndarray,
    edge_vals: np.ndarray,
    edge_index: np.ndarray,
    orbitals_row: np.ndarray,
    orbitals_col: np.ndarray,
    n_supercells: int = 1,
    edge_neigh_isc: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    symmetrize_edges: bool = False,
) -> coo_array:
    """Converts an orbital matrix from node and edges array to scipy coo.

    Conversions to any other sparse structure can be done once we've got the coo array.

    Parameters
    ----------
    node_vals
        Flat array containing the values of the node blocks.
        The order of the values is first by node index, then row then column.
    edge_vals
        Flat array containing the values of the edge blocks.
        The order of the values is first by edge index, then row then column.
    edge_index
        Array of shape (2, n_edges) containing the indices of the atoms
        that participate in each edge.
    orbitals_row / orbitals_col
        Array of shape (n_nodes, ) containing the number of orbitals for each atom.
        For contracted basis sets, rows represent the original basis, and columns the
        contracted basis. For non-contracted basis sets, both are equal.
    n_supercells
        Number of auxiliary supercells.
    edge_neigh_isc
        Array of shape (n_edges, ) containing the supercell index of the second atom
        in each edge with respect to the first atom.
        If not provided, all interactions are assumed to be in the unit cell.
    threshold
        Matrix elements with a value below this number are set to 0.
    symmetrize_edges
        whether for each edge only one direction is provided. The edge block for the
        opposite direction is then created as the transpose.
    """

    def _init_coo(data, rows, cols, shape):
        return coo_array((data, (rows, cols)), shape)

    return _nodes_and_edges_to_coo(
        concatenate=np.concatenate,
        init_coo=_init_coo,
        node_vals=node_vals,
        edge_vals=edge_vals,
        edge_index=edge_index,
        orbitals_row=orbitals_row,
        orbitals_col=orbitals_col,
        n_supercells=n_supercells,
        edge_neigh_isc=edge_neigh_isc,
        threshold=threshold,
        symmetrize_edges=symmetrize_edges,
    )


@converter(Formats.SCIPY_CSR, Formats.SISL)
def csr_to_sisl_sparse_orbital(
    csr: csr_array,
    geometry: sisl.Geometry,
    sp_class: Type[SparseOrbital] = SparseOrbital,
) -> SparseOrbital:
    """Converts a scipy CSR array to a sisl sparse orbital matrix."""
    return sp_class.fromsp(geometry, csr)


@converter(Formats.NODESEDGES, Formats.SISL)
def nodes_and_edges_to_sparse_orbital(
    node_vals: np.ndarray,
    edge_vals: np.ndarray,
    edge_index: np.ndarray,
    geometry: sisl.Geometry,
    sp_class: Type[SparseOrbital] = SparseOrbital,
    edge_neigh_isc: Optional[np.ndarray] = None,
    threshold: float = 1e-8,
    symmetrize_edges: bool = False,
) -> SparseOrbital:
    new_csr = conversions.get_converter(Formats.NODESEDGES, Formats.SCIPY_CSR)(
        node_vals=node_vals,
        edge_vals=edge_vals,
        edge_index=edge_index,
        edge_neigh_isc=edge_neigh_isc,
        orbitals=geometry.orbitals,
        n_supercells=geometry.n_s,
        threshold=threshold,
        symmetrize_edges=symmetrize_edges,
    )

    new_csr.indices = new_csr.indices.astype(np.int32)
    new_csr.indptr = new_csr.indptr.astype(np.int32)

    return csr_to_sisl_sparse_orbital(new_csr, geometry=geometry, sp_class=sp_class)
