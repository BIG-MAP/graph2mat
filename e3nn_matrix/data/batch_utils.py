from typing import Any, Type, Union, Dict, Iterator

import numpy as np
import torch
import sisl

from .periodic_table import AtomicTableWithEdges

def batch_to_sparse_orbital(
    batch: Any,
    z_table: AtomicTableWithEdges,
    matrix_cls: Type[sisl.SparseOrbital],
    prediction: Union[Dict,None]=None,
    symmetric_matrix: bool=False,
    add_atomic_contribution: bool=False,
    ) -> Iterator[sisl.SparseOrbital]:
    """
    Convert a batch of data points to a sparse matrix representation for each configuration
    Parameters
    ----------
        batch : Any
        z_table : AtomicTableWithEdges
        matrix_cls : Type[sisl.SparseOrbital] Class used for the output matrix
        prediction : Union[Dict, None]
            If not None, use entries with keys 'node_labels' and 'edge_labels' for the matrix elements
        symmetric_matrix : bool
            If true, the labels only correspond to 'half' of the symmetric matrix
        add_atomic_contribution : bool
            If true, add the isolated atom contributions to the density
    Yields
    ------
        Sparse orbital matrix (of class matrix_cls) for each configuration in batch
    """

    if prediction is not None:
        # Pointer arrays to understand where the data for each structure starts in the batch. 
        atom_ptr = batch.ptr.numpy(force=True)
        edge_ptr = np.zeros_like(atom_ptr)
        np.cumsum(batch.n_edges.numpy(force=True), out=edge_ptr[1:])

        # Types for both atoms and edges.
        atom_types = batch.atom_types.numpy(force=True)
        edge_types = batch.edge_types.numpy(force=True)

        # Get the values for the node blocks and the pointer to the start of each block.
        node_labels_ptr = z_table.atom_block_pointer(atom_types)

        # Get the values for the edge blocks and the pointer to the start of each block.
        if symmetric_matrix:
            edge_types = edge_types[::2]
            edge_ptr = edge_ptr // 2

        edge_labels_ptr = z_table.edge_block_pointer(edge_types)


        # Loop through structures in the batch
        for i, (atom_start, edge_start) in enumerate(zip(atom_ptr[:-1], edge_ptr[:-1])):
            atom_end = atom_ptr[i + 1]
            edge_end = edge_ptr[i + 1]

            # Get one example from batch
            example = batch.get_example(i)
            # Override node and edge labels if predictions are given
            if prediction is not None:
                node_labels = prediction['node_labels'].numpy(force=True)
                edge_labels = prediction['edge_labels'].numpy(force=True)
                new_atom_label = node_labels[node_labels_ptr[atom_start]:node_labels_ptr[atom_end]]
                new_edge_label = edge_labels[edge_labels_ptr[edge_start]:edge_labels_ptr[edge_end]]
                np.testing.assert_array_equal(new_atom_label.shape, example.atom_labels.numpy(force=True).shape)
                np.testing.assert_array_equal(new_edge_label.shape, example.edge_labels.numpy(force=True).shape)
                example.atom_labels = torch.tensor(new_atom_label)
                example.edge_labels = torch.tensor(new_edge_label)
            # Construct the matrix
            matrix = example.to_sparse_orbital_matrix(z_table, matrix_cls, symmetric_matrix, add_atomic_contribution)
            yield matrix
    else:
        for i in range(batch.num_graphs):
            example = batch.get_example(i)
            yield example.to_sparse_orbital_matrix(z_table, matrix_cls, symmetric_matrix, add_atomic_contribution)