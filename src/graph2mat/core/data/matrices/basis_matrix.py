from __future__ import annotations

from typing import Dict, Tuple, Union

from dataclasses import dataclass
import numpy as np

from ..table import BasisTableWithEdges

BasisCount = np.ndarray  # [num_points]


@dataclass
class BasisMatrix:
    """Container to store the raw matrices as a dictionary of blocks.

    The matrices are stored in this format in `BasisConfiguration`, until
    they are converted to flat arrays for training in `BasisMatrixData`.

    As a user you probably don't need to initialize these matrices explicitly,
    they are initialized appropiately when initializing a `BasisConfiguration`
    object using the `OrbitalConfiguration.new` method.
    """

    #: Dictionary containing the blocks of the matrix. The keys are tuples
    #: `(i, j_uc, sc_j)` where `ì` and `j_uc` are the point indices of the block
    #: (in the unit cell) and `sc_j` is the index of the neighboring cell where
    # `j` is located, e.g. 0 for a unit cell connection.
    block_dict: Dict[Tuple[int, int, int], np.ndarray]
    # Size of the auxiliary supercell. This is the number of cells required
    # in each direction to account for all the interactions of the points in
    # the unit cell. If the point distribution is not periodic, this will
    # always be [1,1,1].
    nsc: np.ndarray
    # Array containing the number of basis functions for each point
    basis_count: BasisCount

    def to_flat_nodes_and_edges(
        self,
        edge_index: np.ndarray,
        edge_sc_shifts: np.ndarray,
        points_order: Union[np.ndarray, None] = None,
        basis_table: Union[BasisTableWithEdges, None] = None,
        point_types: Union[np.ndarray, None] = None,
        sub_point_matrix: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the matrix to a flat representation of the nodes and edges values.

        This representation might be useful for training a neural network, where you will
        want to compare the output array to the target array. If you have flat arrays instead
        of block dicts you can easily compare the whole matrix at once.

        Parameters
        ----------
        edge_index : np.ndarray
            Array of shape [2, n_edges] containing the indices of the points that form each edge.
        edge_sc_shifts : np.ndarray
            Array of shape [n_edges, 3] containing the supercell shifts of each edge. That is, if
            an edge is from point i to a periodic image of point j in the [1,0,0] cell, edge_sc_shifts
            for this edge should be [1,0,0]. Note that for the reverse edge (if there is one), the shift
            will be [-1,0,0].
        points_order : Union[np.ndarray, None], optional
            Array of shape [n_points] containing the order in which the points should be flattened.
            If None, the order will simply be determined by their index.
        basis_table : Union[BasisTableWithEdges, None], optional
            Table containing the types of the points. Only needed if sub_point_matrix is True.
        """

        if points_order is None:
            order = np.arange(len(self.basis_count))
        else:
            order = points_order
            assert len(order) == len(self.basis_count)

        if sub_point_matrix:
            assert basis_table is not None and point_types is not None
            point_matrices = self.get_point_matrices(basis_table)
            blocks = [
                (self.block_dict[i, i, 0] - point_matrices[point_types[i]]).flatten()
                for i in order
                if self.basis_count[i] > 0
            ]
        else:
            blocks = [
                self.block_dict[i, i, 0].flatten()
                for i in order
                if self.basis_count[i] > 0
            ]

        node_values = np.concatenate(blocks)

        assert edge_index.shape[0] == 2, "edge_index is assumed to be [2, n_edges]"
        blocks = [
            self.block_dict[edge[0], edge[1], sc_shift].flatten()
            for edge, sc_shift in zip(edge_index.transpose(), edge_sc_shifts)
            if self.basis_count[edge[0]] > 0 and self.basis_count[edge[1]] > 0
        ]

        edge_values = np.concatenate(blocks)

        return node_values, edge_values

    def get_point_matrices(
        self, basis_table: BasisTableWithEdges
    ) -> Dict[int, np.ndarray]:
        """This method should implement a way of retreiving the sub-matrices of each individual point.

        This is, the matrix that the point would have if it was the only point in the system. This matrix
        will depend on the type of the point, so the basis_table needs to be provided.

        The user might choose to subtract this matrix from the block_dict matrix during training, so that
        the model only learns the interactions between points.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a way of retreiving point matrices."
        )
