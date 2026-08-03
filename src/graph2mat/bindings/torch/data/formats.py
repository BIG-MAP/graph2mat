"""Extensions to the registered formats/conversions for ``torch`` tensors."""

from typing import Optional

import torch
import numpy as np

from graph2mat.core.data import conversions, Formats
from graph2mat.core.data.sparse import _nodes_and_edges_to_coo

Formats.add_alias(Formats.TORCH, torch.Tensor)
Formats.add_alias(Formats.TORCH_COO, torch.sparse_coo_tensor)
Formats.add_alias(Formats.TORCH_CSR, torch.sparse_csr_tensor)

converter = conversions.converter


@converter
def _coo_to_csr(coo: torch.sparse_coo_tensor) -> torch.sparse_csr_tensor:
    return coo.to_sparse_csr()


@converter
def _coo_to_dense(coo: torch.sparse_coo_tensor) -> torch.Tensor:
    return coo.to_dense()


@converter
def _csr_to_coo(csr: torch.sparse_csr_tensor) -> torch.sparse_coo_tensor:
    return csr.to_sparse_coo()


@converter
def _csr_to_dense(csr: torch.sparse_csr_tensor) -> torch.Tensor:
    return csr.to_dense()


@converter
def _torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy(force=True)


@converter
def _numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    if issubclass(array.dtype.type, float):
        return torch.tensor(array, dtype=torch.get_default_dtype())
    else:
        return torch.tensor(array)


@converter(Formats.TORCH_NODESEDGES, Formats.TORCH_COO)
def nodes_and_edges_to_coo(
    node_vals: torch.Tensor,
    edge_vals: torch.Tensor,
    edge_index: torch.Tensor,
    orbitals_row: torch.Tensor,
    orbitals_col: torch.Tensor,
    n_supercells: int = 1,
    edge_neigh_isc: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    symmetrize_edges: bool = False,
) -> torch.sparse_coo_tensor:
    """Converts an orbital matrix from node and edges array to torch coo.

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
    orbitals
        Array of shape (n_nodes, ) containing the number of orbitals for each atom.
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
        return torch.sparse_coo_tensor(
            torch.stack([torch.tensor(rows), torch.tensor(cols)]), data, shape
        )

    return _nodes_and_edges_to_coo(
        concatenate=torch.concatenate,
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
