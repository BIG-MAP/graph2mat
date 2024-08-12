"""Core of the data processing.

Managing sparse matrix data in conjunction with graphs is not trivial:

- **Matrices are sparse**.
- Matrices are in a basis which is centered around the points in the graph. Therefore
  **elements of the matrix correspond to nodes or edges of the graph**.
- Each point might have more than one basis function, therefore **the matrix is divided
  in blocks (not just single elements)** that correspond to nodes or edges of the graph.
- Different point types might have different basis size, which makes **the different
  blocks in the matrix have different shapes**.
- **The different block sizes and the sparsity of the matrices supose and extra
  challenge when batching** examples for machine learning.

This module implements `BasisMatrixData`, a class that

The tools in this submodule are agnostic to the machine learning framework
of choice, and they are based purely on `numpy`, with the extra dependency on `sisl`
to handle the sparse matrices. The `sisl` dependency could eventually be lift off
if needed.
"""
from __future__ import annotations

from typing import (
    Optional,
    Tuple,
    Union,
    Dict,
    Any,
    Callable,
    Sequence,
    Generator,
    List,
)
from functools import cached_property
from pathlib import Path
import dataclasses
from copy import copy
import warnings

import sisl
import numpy as np

import torch

try:
    from torch_geometric.data import Batch
except ImportError:

    class Batch:
        pass


from .basis import get_change_of_basis
from .neighborhood import get_neighborhood
from .configuration import BasisConfiguration, OrbitalConfiguration, PhysicsMatrixType
from .sparse import nodes_and_edges_to_sparse_orbital, nodes_and_edges_to_csr
from .table import BasisTableWithEdges
from .node_feats import OneHotZ

__all__ = ["MatrixDataProcessor", "BasisMatrixData"]


@dataclasses.dataclass(frozen=True)
class MatrixDataProcessor:
    """Data structure that contains all the parameters to interface the real world with the ML models.

    Contains all the objects and implements all the logic (using these objects but never modifying them)
    to convert:
      - A "real world" object (a structure, a path to a structure, a path to a run, etc.) into
        the inputs for the model.
      - The outputs of the model into a "real world" object (a matrix).

    Ideally, any processing that requires the attributes of the data processor (basis_table, symmetric_matrix, etc.)
    should be implemented inside this class so that implementations are not sensitive to small changes
    like the name of the attributes.

    Therefore, every model should have associated a MatrixDataProcessor object to ensure that the
    input is correctly preprocessed and the output is interpreted correctly.

    This data processor is agnostic to the framework of the model (e.g. pytorch) and the processing
    is divided in small functions so that it can be easily reused.

    Parameters
    ------------
    basis_table :
        Table containing all the basis information.
    symmetric_matrix :
        Whether the matrix is symmetric or not.
    sub_point_matrix :
        Whether the isolated point matrix is subtracted from the point labels.
        That would mean that the model is learning a delta with respect to the
        case where all points are isolated.
    out_matrix :
        Type of matrix to output. If None, the matrix is output as a `scipy` CSR matrix.
    """

    basis_table: BasisTableWithEdges
    symmetric_matrix: bool = False
    sub_point_matrix: bool = True
    out_matrix: Optional[PhysicsMatrixType] = None
    node_attr_getters: List[Any] = dataclasses.field(default_factory=list)

    def copy(self, **kwargs):
        """Create a copy of the object with the given attributes replaced."""
        return dataclasses.replace(self, **kwargs)

    @cached_property
    def matrix_cls(self):
        return {
            "density_matrix": sisl.DensityMatrix,
            "energy_density_matrix": sisl.EnergyDensityMatrix,
            "hamiltonian": sisl.Hamiltonian,
            None: None,
        }[self.out_matrix]

    def get_config_kwargs(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, (str, Path)):
            kwargs = {"out_matrix": self.out_matrix}
            if hasattr(self.basis_table, "atoms"):
                kwargs["basis"] = self.basis_table.atoms
            return kwargs
        else:
            return {}

    def torch_predict(self, torch_model, geometry: sisl.Geometry):
        import torch
        from torch_geometric.loader import DataLoader

        from graph2mat.bindings.torch import TorchBasisMatrixData

        with torch.no_grad():
            # USE THE MODEL
            # First, we need to process the input data, to get inputs as the model expects.
            input_data = TorchBasisMatrixData.new(
                geometry, data_processor=self, labels=False
            )

            input_data["ptr"] = torch.tensor([0, len(input_data["point_types"])])
            input_data["batch"] = torch.tensor([0])

            # Then, we run the model.
            out = torch_model(input_data)

            # And finally, we convert the output to a matrix.
            matrix = self.matrix_from_data(input_data, predictions=out)

        return matrix

    def matrix_from_data(
        self,
        data: BasisMatrixData,
        predictions: Optional[Dict] = None,
        threshold: float = 1e-8,
        is_batch: Optional[bool] = None,
    ):
        """Converts a BasisMatrixData object into a matrix.

        It takes into account the matrix class associated to the data processor
        to return the corresponding matrix type.

        It can also convert batches.

        Parameters
        ------------
        data:
            The data to convert.
        predictions:
            Predictions for the matrix labels, with the keys:
                - node_labels: matrix elements that belong to node blocks.
                - edge_labels: matrix elements that belong to edge blocks.

            If None, the labels from the data object are used.
        threshold:
            Elements with a value below this number will be considered 0.
        is_batch:
            Whether the data is a batch or not.

            If None, it will be considered a batch if it is an instance of
            `torch_geometric`'s `Batch`.

        Returns
        ---------
        A matrix if data is not a batch, a tuple of matrices if it is a batch.

        See Also
        ---------
        yield_from_batch: The more explicit option for batches, which returns a generator.
        """

        if is_batch is None:
            is_batch = isinstance(data, Batch)
        if is_batch:
            return tuple(
                self.yield_from_batch(data, predictions, threshold, as_matrix=True)
            )

        if predictions is not None:
            data = copy(data)
            data.point_labels = predictions["node_labels"]
            data.edge_labels = predictions["edge_labels"]

        if self.matrix_cls is None:
            return data.to_csr(threshold=threshold)
        else:
            return data.to_sparse_orbital_matrix(threshold=threshold)

    def yield_from_batch(
        self,
        data: BasisMatrixData,
        predictions: Optional[Dict] = None,
        threshold: float = 1e-8,
        as_matrix: bool = False,
    ) -> Generator:
        """Yields matrices from a batch.

        It takes into account the matrix class associated to the data processor
        to return the corresponding matrix type.

        Parameters
        ------------
        data:
            The batched data.
        predictions:
            Predictions for the matrix labels, with the keys:
                - node_labels: matrix elements that belong to node blocks.
                - edge_labels: matrix elements that belong to edge blocks.

            If None, the labels from the data object are used.
        threshold:
            Elements with a value below this number will be considered 0.
        as_matrix:
            Whether to return a matrix or a BasisMatrixData object.

        See Also
        ---------
        matrix_from_data: The method used to convert data to matrices, which can
            also be called with a batch.
        """
        if predictions is None:
            for i in range(data.num_graphs):
                example = data.get_example(i)
                if as_matrix:
                    yield self.matrix_from_data(example, threshold=threshold)
                else:
                    yield example
        else:
            arrays = data.numpy_arrays()

            # Pointer arrays to understand where the data for each structure starts in the batch.
            atom_ptr = arrays.ptr
            edge_ptr = np.zeros_like(atom_ptr)
            np.cumsum(arrays.n_edges, out=edge_ptr[1:])

            # Types for both atoms and edges.
            point_types = arrays.point_types
            edge_types = arrays.edge_types

            # Get the values for the node blocks and the pointer to the start of each block.
            node_labels_ptr = self.basis_table.point_block_pointer(point_types)

            # Get the values for the edge blocks and the pointer to the start of each block.
            if self.symmetric_matrix:
                edge_types = edge_types[::2]
                edge_ptr = edge_ptr // 2

            edge_labels_ptr = self.basis_table.edge_block_pointer(edge_types)

            # Loop through structures in the batch
            for i, (atom_start, edge_start) in enumerate(
                zip(atom_ptr[:-1], edge_ptr[:-1])
            ):
                atom_end = atom_ptr[i + 1]
                edge_end = edge_ptr[i + 1]

                # Get one example from batch
                example = data.get_example(i)
                # Override node and edge labels if predictions are given
                node_labels = predictions["node_labels"]
                edge_labels = predictions["edge_labels"]
                new_atom_label = node_labels[
                    node_labels_ptr[atom_start] : node_labels_ptr[atom_end]
                ]
                new_edge_label = edge_labels[
                    edge_labels_ptr[edge_start] : edge_labels_ptr[edge_end]
                ]

                if getattr(example, "point_labels", None) is not None:
                    assert len(new_atom_label) == len(example.point_labels)
                if getattr(example, "edge_labels", None) is not None:
                    assert len(new_edge_label) == len(example.edge_labels)

                example.point_labels = new_atom_label
                example.edge_labels = new_edge_label

                if as_matrix:
                    yield self.matrix_from_data(example, threshold=threshold)
                else:
                    yield example

    def compute_metrics(
        self,
        output: dict,
        input: BasisMatrixData,
        metrics: Union[Sequence["OrbitalMatrixMetric"], None] = None,
    ) -> dict:
        """Computes the metrics for a given output and input.

        Parameters
        ------------
        output: dict
            Output of the model, as it comes out of it.
        input: BasisMatrixData
            The input that was passed to the model.
        metrics: Sequence[OrbitalMatrixMetric], optional
            Metrics to compute. If None, all known metrics are computed.

        Returns
        ---------
        dict
            Dictionary where keys are the names of the metrics and values are their values.
        """
        from .metrics import OrbitalMatrixMetric

        if metrics is None:
            metrics = [
                metric_cls for metric_cls in OrbitalMatrixMetric.__subclasses__()
            ]

        input_arrays = input.numpy_arrays()

        metrics_values = [
            metric(
                nodes_pred=input.ensure_numpy(output["node_labels"]),
                nodes_ref=input_arrays.point_labels,
                edges_pred=input.ensure_numpy(output["edge_labels"]),
                edges_ref=input_arrays.edge_labels,
                batch=input,
                basis_table=self.basis_table,
                config_resolved=False,
                symmetric_matrix=self.symmetric_matrix,
            )[0]
            for metric in metrics
        ]

        return {
            metric.__name__: float(value)
            for metric, value in zip(metrics, metrics_values)
        }

    def add_basis_to_geometry(self, geometry: sisl.Geometry) -> sisl.Geometry:
        """Returns a copy of the geometry with the basis of this processor added to it.

        It works by replacing an atom with atomic number Z in the geometry with the atom
        with the same Z in the basis table.
        """

        new_geometry = geometry.copy()

        for atom in geometry.atoms.atom:
            for basis_atom in self.basis_table.atoms:
                if basis_atom.Z == atom.Z:
                    break
            else:
                raise ValueError(f"Couldn't find atom {atom} in the basis")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_geometry.atoms.replace_atom(atom, basis_atom)

        return new_geometry

    def get_point_block_rtps(self):
        from e3nn import o3

        basis = self.basis_table.basis

        symmetry = "ij=ji" if self.symmetric_matrix else "ij"
        return tuple(
            o3.ReducedTensorProducts(symmetry, i=pb.e3nn_irreps, j=pb.e3nn_irreps)
            for pb in basis
        )

    def get_edge_block_rtps(self):
        from e3nn import o3

        basis = self.basis_table.basis

        return tuple(
            o3.ReducedTensorProducts(
                "ij", i=basis[i].e3nn_irreps, j=basis[j].e3nn_irreps
            )
            for i, j in self.basis_table.edge_type_to_point_types
        )

    def irreps_from_data(self, data):
        """Converts the matrix data into irreps."""
        from e3nn import o3

        if not self.symmetric_matrix:
            raise NotImplementedError("Only implemented for symmetric matrices")

        point_labels = data.point_labels
        arrays = data.numpy_arrays()

        point_types = arrays["point_types"]
        point_pointers = self.basis_table.point_block_pointer(point_types)

        reduced_tensor_products = self.get_point_block_rtps()

        point_irreps = []
        point_values = []
        for i in range(len(point_types)):
            point_type = point_types[i]

            rtp = reduced_tensor_products[point_type]
            point_irreps.extend(list(rtp.irreps_out))

            block = point_labels[point_pointers[i] : point_pointers[i + 1]]

            shape = self.basis_table.point_block_shape[:, point_type]
            block = block.reshape(tuple(shape))

            irreps_values = torch.einsum("zij, ij-> z", rtp.change_of_basis, block)

            point_values.append(irreps_values)

        edge_labels = data.edge_labels
        edge_types = arrays["edge_types"][::2]
        edge_pointers = self.basis_table.edge_block_pointer(edge_types)

        reduced_tensor_products = self.get_edge_block_rtps()

        edge_irreps = []
        edge_values = []
        for i in range(len(edge_types)):
            edge_type = edge_types[i]

            rtp = reduced_tensor_products[edge_type]
            edge_irreps.extend(list(rtp.irreps_out))

            block = edge_labels[edge_pointers[i] : edge_pointers[i + 1]]

            shape = self.basis_table.edge_block_shape[:, edge_type]
            block = block.reshape(tuple(shape))

            irreps_values = torch.einsum("zij, ij-> z", rtp.change_of_basis, block)

            edge_values.append(irreps_values)

        return {
            "node_labels": torch.concatenate(point_values),
            "node_irreps": o3.Irreps(point_irreps),
            "edge_labels": torch.concatenate(edge_values),
            "edge_irreps": o3.Irreps(edge_irreps),
        }

    @staticmethod
    def sort_edge_index(
        edge_index: np.ndarray,
        sc_shifts: np.ndarray,
        shifts: np.ndarray,
        edge_types: np.ndarray,
        isc_off: np.ndarray,
        inplace: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the sorted edge indices.

        Edges are much easier to manipulate by the block producing routines if they are ordered properly.

        This function orders edges in a way that both directions of the same edge come consecutively.
        It also always puts first the interaction (lowest point type, highest point type), that is the
        one with positive edge type.

        For the unit cell, the connection in different directions is simple to understand, as it's just
        a permutation of the points. I.e. edges (i, j) and (j, i) are the same connection in opposite directions.
        However, for connections between supercells (if there are periodic conditions), this condition is not
        enough. The supercell shift of one direction must be the negative of the other direction. I.e.
        only edges between (i, j, x, y, z) and (j, i, -x, -y, -z) are the same connection in opposite directions.
        It is also important to notice that in the supercell connections i and j can be the same index.

        Parameters
        -------------
        edge_index: np.ndarray of shape (2, n_edges)
            Pair of point indices for each edge.
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
        # points in the connection is sorted by point type. We can easily identify the unordered connections
        # because by convention they are assigned a negative edge type. We don't need to deal with shifts
        # here because they are all 0.
        uc_unique_inter = uc_inter[:, uc_unique_edge]
        uc_unique_edge_types = edge_types[uc_edge][uc_unique_edge]
        uc_unordered = uc_unique_edge_types < 0
        uc_unique_inter[:, uc_unordered] = uc_unique_inter[::-1, uc_unordered]

        # Manipulate supercell connections: Get the unique interactions and make sure the pair of
        # points in the connection is sorted by point type, doing the same as in the unit cell case.
        # However, in this case we care about the shifts because (1) we need to select the ones corresponding
        # to the selected connections and (2) we need to flip their direction for the connections that we re-sort.
        sc_unique_inter = edge_index[:, sc_unique_edge]
        sc_unique_inter_sc_shift = sc_shifts[:, sc_unique_edge]
        sc_unique_inter_shift = shifts[:, sc_unique_edge]
        sc_unique_edge_types = edge_types[sc_unique_edge]

        # Order supercell connections.
        sc_unordered = edge_types[sc_unique_edge] < 0
        sc_unique_inter[:, sc_unordered] = sc_unique_inter[::-1, sc_unordered]
        sc_unique_inter_sc_shift[:, sc_unordered] = -sc_unique_inter_sc_shift[
            :, sc_unordered
        ]
        sc_unique_inter_shift[:, sc_unordered] = -sc_unique_inter_shift[:, sc_unordered]

        # Stack both unit cell and supercell connections
        unique_interactions = np.hstack([uc_unique_inter, sc_unique_inter])
        unique_sc_shifts = np.hstack(
            [np.zeros((3, uc_unique_inter.shape[1])), sc_unique_inter_sc_shift]
        )
        unique_shifts = np.hstack(
            [np.zeros((3, uc_unique_inter.shape[1])), sc_unique_inter_shift]
        )
        unique_edge_types = abs(
            np.concatenate([uc_unique_edge_types, sc_unique_edge_types])
        )

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
        edge_types[1::2] = -edge_types[::2]

        # And also shifts and supercell shifts
        shifts[:, ::2] = unique_shifts
        shifts[:, 1::2] = -unique_shifts

        sc_shifts[:, ::2] = unique_sc_shifts
        sc_shifts[:, 1::2] = -unique_sc_shifts

        return edge_index, sc_shifts, shifts, edge_types

    def cartesian_to_basis(
        self,
        array: Union[np.ndarray, torch.Tensor],
        process_cob_array: Optional[Callable] = None,
    ):
        cob = self.basis_table.change_of_basis
        if process_cob_array is not None:
            cob = process_cob_array("change_of_basis", cob)

        return array @ cob.T

    def basis_to_cartesian(
        self,
        array: Union[np.ndarray, torch.Tensor],
        process_cob_array: Optional[Callable] = None,
    ):
        cob = self.basis_table.change_of_basis_inv
        if process_cob_array is not None:
            cob = process_cob_array("change_of_basis_inv", cob)
        return array @ cob.T

    def get_point_types(self, config: BasisConfiguration) -> np.ndarray:
        """Returns the type (index in the basis table) of each point in the configuration."""
        return self.basis_table.types_to_indices(config.point_types)

    def get_cutoff(self, point_types: np.ndarray) -> Union[float, np.ndarray]:
        """Returns the cutoff radius.

        Parameters
        ----------
        point_types : np.ndarray of shape (n_points,)
            Type of each point (index in the basis table) in the configuration.

        Returns
        -------
        float or np.ndarray of shape (n_points,)
            The cutoff radius might be a single number if all points have the same cutoff radius,
            or an array with the cutoff radius of each point.

            If each point has its own radius, one might find an edge between i and j if
            dist(i -> j) is smaller than cutoff_i + cutoff_j.

            If the cutoff radius is a single number, edges are found if dist(i -> j) is smaller than cutoff.
        """

        if isinstance(self.basis_table.R, float):
            return self.basis_table.R * 2
        else:
            return self.basis_table.R[point_types]

    def get_nlabels_per_edge_type(self, edge_types: np.ndarray) -> np.ndarray:
        """Returns the number of labels for each edge type in a given matrix.

        It takes into account whether the matrix is symmetric or not (if it is,
        the number of labels for each edge type is divided by 2).

        Returns
        -------
        edge_type_nlabels : np.ndarray of shape (n_edge_types,)
            Number of labels required for each edge type.
        """
        unique_edge_types, counts = np.unique(abs(edge_types), return_counts=True)
        if self.symmetric_matrix:
            counts = counts / 2

        edge_type_nlabels = np.zeros(
            self.basis_table.edge_type[-1, -1] + 1, dtype=np.int64
        )
        edge_type_nlabels[unique_edge_types] = (
            self.basis_table.edge_block_size[unique_edge_types] * counts
        )

        return edge_type_nlabels

    def get_labels_from_types_and_edges(
        self,
        config: BasisConfiguration,
        point_types: np.ndarray,
        edge_index: np.ndarray,
        neigh_isc: np.ndarray,
    ) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        """Once point types and edges have been determined, one can call this function to get the labels of the matrix.

        Parameters
        ----------
        config : BasisConfiguration
            The configuration from which the labels will be extracted.
        point_types : np.ndarray of shape (n_points,)
            Type of each point (index in the basis table) in the configuration.
        edge_index : np.ndarray of shape (2, n_edges)
            Array with point pairs (their index in the configuration) that form an edge.
        neigh_isc : np.ndarray of shape (n_edges,)
            Array with the index of the supercell shift of the second point of each edge.

        Returns
        -------
        point_labels : np.ndarray of shape (n_point_labels, )
            Array with the flattened labels for each point block in the  in the configuration.
        edge_labels : np.ndarray of shape (n_edge_labels, )
            Array with the flattened labels for each edge block in the configuration.
        """

        if config.matrix is not None:
            if self.symmetric_matrix:
                needed_edge_blocks = edge_index[:, ::2]  # only unique interactions
                needed_neigh_isc = neigh_isc[::2]
            else:
                needed_edge_blocks = edge_index
                needed_neigh_isc = neigh_isc

            point_labels, edge_labels = config.matrix.to_flat_nodes_and_edges(
                edge_index=needed_edge_blocks,
                edge_sc_shifts=needed_neigh_isc,
                point_types=point_types,
                basis_table=self.basis_table,
                sub_point_matrix=self.sub_point_matrix,
            )
        else:
            # We are most likely in predict mode.
            # We have no matrix data
            point_labels = edge_labels = None

        return point_labels, edge_labels

    def get_node_attrs(self, config: BasisConfiguration) -> np.ndarray:
        """Returns the initial features of nodes."""
        node_attr_getters = self.node_attr_getters
        if len(node_attr_getters) == 0:
            node_attr_getters = [OneHotZ]
        return np.concatenate(
            [getter(config, self) for getter in node_attr_getters], axis=1
        )

    def one_hot_encode(self, point_types: np.ndarray) -> np.ndarray:
        """One hot encodes a vector of point types.

        It takes into account the number of different point types in the basis table.

        Parameters
        ----------
        point_types : np.ndarray of shape (n_points,)
            Array of point types (their index in the basis table).

        Returns
        -------
        np.ndarray of shape (n_points, n_classes)
            One hot encoded array of point types.
        """
        num_classes = len(self.basis_table)

        return np.eye(num_classes)[point_types]

    def labels_to_csr(
        self,
        data: dict[str, np.ndarray],
        coords_cartesian: bool = False,
        threshold: float = 1e-8,
    ):
        # Get all the arrays that we need.
        node_labels = data["point_labels"]
        edge_labels = data["edge_labels"]

        point_types = data["point_types"]
        edge_types = data["edge_types"]

        edge_index = data["edge_index"]
        neigh_isc = data["neigh_isc"]

        positions = data["positions"]

        if not coords_cartesian:
            # Positions need to be converted to cartesian coordinates
            positions = self.basis_to_cartesian(positions)

        nsc = data["nsc"].squeeze()

        # Get the values for the node blocks and the pointer to the start of each block.
        node_labels_ptr = self.basis_table.point_block_pointer(point_types)

        # Add back atomic contributions to the node blocks in case they were removed
        if self.sub_point_matrix:
            assert self.basis_table.point_matrix is not None, "Point matrices"
            node_labels = node_labels + np.concatenate(
                [
                    self.basis_table.point_matrix[atom_type].ravel()
                    for atom_type in point_types
                ]
            )

        # Get the values for the edge blocks and the pointer to the start of each block.
        if self.symmetric_matrix:
            edge_index = edge_index[:, ::2]
            edge_types = edge_types[::2]
            neigh_isc = neigh_isc[::2]

        edge_labels_ptr = self.basis_table.edge_block_pointer(edge_types)

        n_orbitals = [point.basis_size for point in self.basis_table.basis]
        orbitals = [n_orbitals[at_type] for at_type in point_types]

        nsc = data["nsc"].squeeze()

        # Construct the matrix.
        matrix = nodes_and_edges_to_csr(
            node_vals=node_labels,
            node_ptr=node_labels_ptr,
            edge_vals=edge_labels,
            edge_index=edge_index,
            edge_neigh_isc=neigh_isc,
            edge_ptr=edge_labels_ptr,
            n_supercells=np.prod(nsc),
            orbitals=orbitals,
            symmetrize_edges=self.symmetric_matrix,
            threshold=threshold,
        )

        return matrix

    def labels_to_sparse_orbital(
        self,
        data: dict[str, np.ndarray],
        coords_cartesian: bool = False,
        threshold: float = 1e-8,
    ) -> sisl.SparseOrbital:
        # Get all the arrays that we need.
        node_labels = data["point_labels"]
        edge_labels = data["edge_labels"]

        point_types = data["point_types"]
        edge_types = data["edge_types"]

        edge_index = data["edge_index"]
        neigh_isc = data["neigh_isc"]

        cell = data["cell"]
        positions = data["positions"]

        if not coords_cartesian:
            # Cell and positions need to be converted to cartesian coordinates
            cell = self.basis_to_cartesian(cell)
            positions = self.basis_to_cartesian(positions)

        nsc = data["nsc"].squeeze()

        # Get the values for the node blocks and the pointer to the start of each block.
        node_labels_ptr = self.basis_table.point_block_pointer(point_types)

        # Add back atomic contributions to the node blocks in case they were removed
        if self.sub_point_matrix:
            assert self.basis_table.point_matrix is not None, "Point matrices"
            node_labels = node_labels + np.concatenate(
                [
                    self.basis_table.point_matrix[atom_type].ravel()
                    for atom_type in point_types
                ]
            )

        # Get the values for the edge blocks and the pointer to the start of each block.
        if self.symmetric_matrix:
            edge_index = edge_index[:, ::2]
            edge_types = edge_types[::2]
            neigh_isc = neigh_isc[::2]

        edge_labels_ptr = self.basis_table.edge_block_pointer(edge_types)

        unique_atoms = self.basis_table.get_sisl_atoms()

        geometry = sisl.Geometry(
            positions,
            atoms=[unique_atoms[at_type] for at_type in point_types],
            sc=cell,
        )
        geometry.set_nsc(nsc)

        # Construct the matrix.
        matrix = nodes_and_edges_to_sparse_orbital(
            node_vals=node_labels,
            node_ptr=node_labels_ptr,
            edge_vals=edge_labels,
            edge_index=edge_index,
            edge_neigh_isc=neigh_isc,
            edge_ptr=edge_labels_ptr,
            geometry=geometry,
            sp_class=self.matrix_cls,
            symmetrize_edges=self.symmetric_matrix,
            threshold=threshold,
        )

        # Remove atoms with no basis.
        for i, point_basis in enumerate(self.basis_table.basis):
            if point_basis.basis_size == 0:
                matrix = matrix.remove(unique_atoms[i])

        return matrix


class NumpyArraysProvider:
    """Helper classs to get attributes from the data object making sure they are numpy arrays.

    Some subclasses of `BasisMatrixData` might not use `numpy` arrays to store the data. For example,
    they might use ``torch`` tensors. However, postprocessing tools are usually based on ``numpy``.

    If you get data directly from the data class you are risking that there might be incompatibilities.
    However, if you use the arrays provider, you can be sure that you will receive ``numpy`` arrays.
    """

    def __init__(self, data: BasisMatrixData):
        self.data = data

    def __getitem__(self, key) -> np.ndarray:
        return self.data.ensure_numpy(self.data[key])

    def __getattr__(self, key) -> np.ndarray:
        return self.data.ensure_numpy(self.data[key])

    def __dir__(self):
        return dir(self.data)


class BasisMatrixData:
    """Stores a graph with the preprocessed data for one or multiple configurations.

    The differences between this class and ``BasisConfiguration`` are:

      - This class stores examples as graphs, while ``BasisConfiguration`` just stores the
        raw data.
      - This class might store a batch of examples inside the same graph. Different
        dataset examples are just graph clusters that are not connected with each other.

    This class is the main interface between the data and the models.

    The class accepts positions, cell, and displacements in cartesian coordinates,
    but they are converted to the convention specified by the data processor (e.g. spherical harmonics),
    and stored in this way.

    Parameters
    ------------
    edge_index :
        Shape (2, n_edges).
        Array with point pairs (their index in the configuration) that form an edge.
    neigh_isc :
        Shape (n_edges,).
        Array with the index of the supercell where the second point of each edge
        is located.
        This follows the conventions in ``sisl``
    node_attrs :
        Shape (n_points, n_node_feats).
        Inputs for each point in the configuration.
    positions :
        Shape (n_points, 3).
        Cartesian coordinates of each point in the configuration.
    shifts :
        Shape (n_edges, 3).
        Cartesian shift of the second atom in each edge with respect to its
        image in the primary cell. E.g. if the second atom is in the primary cell,
        the shift will be [0,0,0].
    cell :
        Shape (3,3).
        Lattice vectors of the unit cell in cartesian coordinates.
    nsc :
        Shape (3,).
        Number of auxiliary cells required in each direction to account for
        all neighbor interactions.
    point_labels :
        Shape (n_point_labels,).
        The elements of the target matrix that correspond to interactions
        within the same node. This is flattened to deal with the fact that
        each block might have different shape.

        All values for a given block come consecutively and in row-major order.
    edge_labels :
        Shape (n_edge_labels,).
        The elements of the target matrix that correspond to interactions
        between different nodes. This is flattened to deal with the fact that
        each block might have different shape.

        All values for a given block come consecutively and in row-major order.

        NOTE: These should be sorted by edge type.
    point_types :
        Shape (n_points,).
        The type of each point (index in the basis table).
    edge_types :
        Shape (n_edges,).
        The type of each edge as defined by the basis table.
    edge_type_nlabels :
        Shape (n_edge_types,).
        Edge labels are sorted by edge type. This array contains the number of
        labels for each edge type.
    data_processor :
        Data processor associated to this data.
    metadata :
        Contains any extra metadata that might be useful for the model or to
        postprocess outputs, for example.
    """

    #: Sometimes it is useful to know explicitly which keys are node attributes
    #: The list is stored in this variable.
    _node_attr_keys: Sequence[str] = ("node_attrs", "positions", "point_types")

    #: Sometimes it is useful to know explicitly which keys are edge attributes
    #: The list is stored in this variable.
    _edge_attr_keys: Sequence[str] = ("edge_types", "shifts", "neigh_isc")

    #: Number of nodes in the configuration
    num_nodes: Optional[int]

    #: Shape (2, n_edges).
    #: Array with point pairs (their index in the configuration) that form an edge.
    edge_index: np.ndarray

    #: Shape (n_edges,).
    #: Array with the index of the supercell where the second point of each edge
    #: is located.
    #: This follows the conventions in ``sisl``
    neigh_isc: np.ndarray

    #: Shape (n_points, n_node_feats).
    #: Inputs for each point in the configuration.
    node_attrs: np.ndarray

    #: Shape (n_points, 3).
    #: Coordinates of each point in the configuration, **in the convention specified
    #: by the data processor (e.g. spherical harmonics)**.
    #: IMPORTANT: This is not necessarily in cartesian coordinates.
    positions: np.ndarray

    #: Shape (n_edges, 3).
    #: Shift of the second atom in each edge with respect to its
    #: image in the primary cell, **in the convention specified
    #: by the data processor (e.g. spherical harmonics)**.
    #: IMPORTANT: This is not necessarily in cartesian coordinates.
    shifts: np.ndarray

    #: Shape (3,3).
    #: Lattice vectors of the unit cell, **in the convention specified
    #: by the data processor (e.g. spherical harmonics)**.
    #: IMPORTANT: This is not necessarily in cartesian coordinates.
    cell: np.ndarray

    #: Total number of auxiliary cells.
    n_supercells: int

    #: Number of auxiliary cells required in each direction to account for
    #: all neighbor interactions.
    nsc: np.ndarray

    #: Shape (n_point_labels,).
    #: The elements of the target matrix that correspond to interactions
    #: within the same node. This is flattened to deal with the fact that
    #: each block might have different shape.
    #:
    #: All values for a given block come consecutively and in row-major order.
    point_labels: np.ndarray

    #: Shape (n_edge_labels,).
    #: The elements of the target matrix that correspond to interactions
    #: between different nodes. This is flattened to deal with the fact that
    #: each block might have different shape.
    #:
    #: All values for a given block come consecutively and in row-major order.
    edge_labels: np.ndarray

    #: Shape (n_points,).
    #: The type of each point (index in the basis table, i.e.
    #: a `BasisTableWithEdges`).
    point_types: np.ndarray

    #: Shape (n_edges,).
    #: The type of each edge as defined by the basis table, i.e.
    #: a `BasisTableWithEdges`.
    edge_types: np.ndarray

    #: Shape (n_edge_types,).
    #: Edge labels are sorted by edge type. This array contains the number of
    #: labels for each edge type.
    edge_type_nlabels: np.ndarray

    labels_point_filter: np.ndarray
    labels_edge_filter: np.ndarray

    #: Contains any extra metadata that might be useful for the model or to
    #: postprocess outputs, for example. It includes the data processor.
    metadata: Dict[str, Any]

    def __init__(
        # All arguments must be optional in order for the get_example method of a batch to work
        self,
        edge_index: Optional[np.ndarray] = None,  # [2, n_edges]
        neigh_isc: Optional[np.ndarray] = None,  # [n_edges,]
        node_attrs: Optional[np.ndarray] = None,  # [n_nodes, n_node_feats]
        positions: Optional[np.ndarray] = None,  # [n_nodes, 3]
        shifts: Optional[np.ndarray] = None,  # [n_edges, 3],
        cell: Optional[np.ndarray] = None,  # [3,3]
        nsc: Optional[np.ndarray] = None,  # [3,]
        point_labels: Optional[np.ndarray] = None,  # [total_point_elements]
        edge_labels: Optional[np.ndarray] = None,  # [total_edge_elements]
        labels_point_filter: Optional[np.ndarray] = None,  # [n_point_labels]
        labels_edge_filter: Optional[np.ndarray] = None,  # [n_edge_labels]
        point_types: Optional[np.ndarray] = None,  # [n_nodes]
        edge_types: Optional[np.ndarray] = None,  # [n_edges]
        edge_type_nlabels: Optional[np.ndarray] = None,  # [n_edge_types]
        data_processor: MatrixDataProcessor = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._data = self._sanitize_data(
            edge_index=edge_index,
            neigh_isc=neigh_isc,
            node_attrs=node_attrs,
            positions=positions,
            shifts=shifts,
            cell=cell,
            nsc=nsc,
            point_labels=point_labels,
            edge_labels=edge_labels,
            labels_point_filter=labels_point_filter,
            labels_edge_filter=labels_edge_filter,
            point_types=point_types,
            edge_types=edge_types,
            edge_type_nlabels=edge_type_nlabels,
            data_processor=data_processor,
            metadata=metadata,
        )

        for k in self._data:
            setattr(self, k, self._data[k])

    def _sanitize_data(
        self,
        edge_index: Optional[np.ndarray] = None,  # [2, n_edges]
        neigh_isc: Optional[np.ndarray] = None,  # [n_edges,]
        node_attrs: Optional[np.ndarray] = None,  # [n_nodes, n_node_feats]
        positions: Optional[np.ndarray] = None,  # [n_nodes, 3]
        shifts: Optional[np.ndarray] = None,  # [n_edges, 3],
        cell: Optional[np.ndarray] = None,  # [3,3]
        nsc: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,  # [total_point_elements]
        edge_labels: Optional[np.ndarray] = None,  # [total_edge_elements]
        labels_point_filter: Optional[np.ndarray] = None,  # [total_point_elements]
        labels_edge_filter: Optional[np.ndarray] = None,  # [total_edge_elements]
        point_types: Optional[np.ndarray] = None,  # [n_nodes]
        edge_types: Optional[np.ndarray] = None,  # [n_edges]
        edge_type_nlabels: Optional[np.ndarray] = None,  # [n_edge_types]
        data_processor: MatrixDataProcessor = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Check shapes
        num_nodes = node_attrs.shape[0] if node_attrs is not None else None

        assert edge_index is None or (
            edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        )
        assert neigh_isc is None or (
            neigh_isc.ndim == 1 and neigh_isc.shape[0] == edge_index.shape[1]
        )
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
            "positions": positions if positions is not None else None,
            "shifts": shifts if shifts is not None else None,
            "cell": cell if cell is not None else None,
            "nsc": nsc.reshape(1, 3) if nsc is not None else None,
            "node_attrs": node_attrs,
            "point_labels": point_labels,
            "edge_labels": edge_labels,
            "point_types": point_types,
            "edge_types": edge_types,
            "edge_type_nlabels": edge_type_nlabels.reshape(1, -1)
            if edge_type_nlabels is not None
            else None,
            "metadata": {
                **(metadata or {}),
                "data_processor": data_processor,
            },
        }

        for k, array in data.items():
            if k not in ["metadata", "num_nodes", "n_edges"] and array is not None:
                data[k] = self.process_input_array(k, array)

        # Because we want an output in the basis of spherical harmonics, we will need to change
        # the basis of the inputs that are cartesian coordinates.
        # See: https://docs.e3nn.org/en/stable/guide/change_of_basis.html
        # We do the change in the inputs (coordinates, which are vectors) because it's much simpler
        # than doing it in the outputs (spherical harmonics with arbitrary l)
        # Matrix with the change of basis to go from cartesian coordinates to spherical harmonics.
        for k in ["positions", "shifts", "cell"]:
            if data[k] is not None:
                data[k] = data_processor.cartesian_to_basis(
                    data[k], process_cob_array=self.process_input_array
                )

        return data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    @classmethod
    def new(
        cls,
        obj: Union[BasisConfiguration, sisl.Geometry, sisl.SparseOrbital, str, Path],
        data_processor: MatrixDataProcessor,
        labels: bool = True,
        **kwargs,
    ) -> "BasisMatrixData":
        """Creates a new basis matrix data object.

        If `obj` is a configuration, the `from_config` method is called.
        Otherwise, we try to first create a configuration from the provided
        arguments and then call the `from_config` method.

        Parameters
        ----------
        obj:
            The object to convert into this class.
        data_processor:
            If `obj` is not a configuration, the data processor
            is needed to understand how to create the basis matrix
            data object.
            In any case, the data processor is needed to convert from
            configuration to basis matrix data ready for models to use
            (e.g. because it contains the basis table).

        See Also
        ----------
        OrbitalConfiguration.new
            The method called to initialize a configuration if `obj` is not a configuration.
        from_config
            The method called to initialize the basis matrix data object.
        """
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, BasisConfiguration):
            config = obj
        else:
            config_kwargs = data_processor.get_config_kwargs(obj)
            config_kwargs.update(kwargs)
            config = OrbitalConfiguration.new(obj, labels=labels, **config_kwargs)

        return cls.from_config(config, data_processor)

    @classmethod
    def from_config(
        cls, config: BasisConfiguration, data_processor: MatrixDataProcessor, nsc=None
    ) -> "BasisMatrixData":
        """Creates a basis matrix data object from a configuration.

        Parameters
        ----------
        config:
            The configuration from which to create the basis matrix data object.
        data_processor:
            The data processor that contains all the information needed to convert
            the configuration into the basis matrix data object. E.g. it contains
            the basis table.
        """
        indices = data_processor.get_point_types(config)
        node_attrs = data_processor.get_node_attrs(config)

        # Search for the neighbors. We use the max radius of each atom as cutoff for looking over neighbors.
        # This means that two atoms ij are neighbors if they have some overlap between their orbitals. That is
        # if distance_ij <= maxR_i + maxR_j. We subtract 0.0001 from the radius to avoid numerical problems that
        # can cause two atoms to be considered neighbors when there is no entry in the sparse matrix.
        edge_index, sc_shifts, shifts = get_neighborhood(
            positions=config.positions,
            cutoff=data_processor.get_cutoff(indices) - 1e-4,  # + 0.2,
            pbc=config.pbc,
            cell=config.cell,
        )

        # Given some supercell offset of all edges, find out what is the minimum number of supercells
        # that we need to describe all interactions.
        sc_shifts = sc_shifts.T

        # Get the number of supercells needed along each direction to account for all interactions
        if nsc is None:
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
        supercell = sisl.Lattice(config.cell, nsc=nsc)

        # Get the edge types
        edge_types = data_processor.basis_table.point_type_to_edge_type(
            indices[edge_index]
        )

        # Sort the edges to make it easier for the reading routines
        data_processor.sort_edge_index(
            edge_index, sc_shifts, shifts.T, edge_types, supercell.isc_off, inplace=True
        )

        # Count the number of labels that this matrix should have per edge type.
        edge_type_nlabels = data_processor.get_nlabels_per_edge_type(
            edge_types=edge_types
        )

        # Then, get the supercell index of each interaction.
        neigh_isc = supercell.isc_off[sc_shifts[0], sc_shifts[1], sc_shifts[2]]

        # Get the flattened labels for the matrix.
        point_labels, edge_labels = data_processor.get_labels_from_types_and_edges(
            config, point_types=indices, edge_index=edge_index, neigh_isc=neigh_isc
        )

        return cls(
            edge_index=edge_index,
            neigh_isc=neigh_isc,
            node_attrs=node_attrs,
            positions=config.positions,
            shifts=shifts,
            cell=config.cell if config.cell is not None else None,
            nsc=supercell.nsc,
            point_labels=point_labels if point_labels is not None else None,
            edge_labels=edge_labels if edge_labels is not None else None,
            point_types=indices,
            edge_types=edge_types,
            edge_type_nlabels=edge_type_nlabels,
            data_processor=data_processor,
            metadata=config.metadata,
        )

    def process_input_array(self, key: str, array: np.ndarray) -> Any:
        """This function might be implemented by subclasses to e.g. convert the array to a torch tensor."""
        return array

    def ensure_numpy(self, array: Any) -> np.ndarray:
        """This function might be implemented by subclasses to convert from their output to numpy arrays.

        This is called by post processing utilities so that they can be sure they are dealing with numpy arrays.
        """
        return np.array(array)

    def numpy_arrays(self) -> NumpyArraysProvider:
        """Returns object that provides data as numpy arrays."""
        return NumpyArraysProvider(self)

    def to_csr(self, threshold: float = 1e-8) -> sisl.SparseOrbital:
        # Get the metadata to process things
        data_processor = self.metadata["data_processor"]

        # Make sure we are dealing with numpy arrays
        arrays = self.numpy_arrays()

        return data_processor.labels_to_csr(arrays, threshold=threshold)

    def to_sparse_orbital_matrix(self, threshold: float = 1e-8) -> sisl.SparseOrbital:
        # Get the metadata to process things
        data_processor = self.metadata["data_processor"]

        # Make sure we are dealing with numpy arrays
        arrays = self.numpy_arrays()

        return data_processor.labels_to_sparse_orbital(arrays, threshold=threshold)

    def node_types_subgraph(self, node_types: np.ndarray) -> "BasisMatrixData":
        """Returns a subgraph with only the nodes of the given types.

        If the BasisMatrixData has labels (i.e. a matrix), this function will
        raise an error because we don't support filtering labels yet.

        Parameters
        ----------
        node_types :
            Array with the node types to keep.
        """
        # Initialize the data dictionary, removing the num_nodes and n_edges keys
        # which should be recomputed on init. Also, pass the data processor as an argument.
        new_data = {**self._data}
        new_data.pop("num_nodes")
        new_data.pop("n_edges")
        new_data["metadata"] = new_data["metadata"].copy()
        new_data["data_processor"] = new_data["metadata"].pop("data_processor", None)

        # Filtering point labels and edge labels is complicated, we don't support it yet
        if "point_labels" in new_data or "edge_labels" in new_data:
            raise ValueError("point_labels and edge_labels are not supported yet")

        # Find the indices of the nodes that belong to the requested types
        mask = np.isin(self.point_types, node_types)
        # And the edge indices for edges between nodes that we will keep
        edge_mask = np.all(mask[self.edge_index], axis=0)

        # Filter node attributes
        for k in self._node_attr_keys:
            if new_data.get(k) is not None:
                new_data[k] = new_data[k][mask]

        # Filter edge indices
        new_data["edge_index"] = new_data["edge_index"][:, edge_mask]

        # Filter edge attributes
        for k in self._edge_attr_keys:
            if new_data.get(k) is not None:
                new_data[k] = new_data[k][edge_mask]

        # Set nlabels to 0 for edge types that are not present anymore
        new_data["edge_type_nlabels"] = copy(new_data["edge_type_nlabels"])
        u_edge_types = abs(new_data["edge_types"]).unique()
        for i in range(new_data["edge_type_nlabels"].shape[1]):
            if i not in u_edge_types:
                new_data["edge_type_nlabels"][:, i] = 0

        return self.__class__(**new_data)
