"""Graph2Mat, the models' skeleton."""

import itertools
import numpy as np
from typing import Sequence, Type, Union, Tuple, List, Dict, TypeVar, Generic, Callable
from types import ModuleType

from ..data import BasisMatrixData
from .matrixblock import MatrixBlock
from ..data.basis import PointBasis

__all__ = ["Graph2Mat"]

# This type will be
ArrayType = TypeVar("ArrayType")


class Graph2Mat(Generic[ArrayType]):
    """Converts a graph to a sparse matrix.

    The matrix that this module computes has variable size, which corresponds
    to the size of the graph. It is built by applying a convolution of its functions
    over the edges and nodes of the graph.

    **High level architecture overview**

    .. image:: /_static/images/Graph2Mat.svg


    **Design concept**

    The module builds the matrix block by block. We define a block as a region of the matrix
    where the rows are all the basis of a given point, and all the columns are the basis of another
    given point. There are then two clearly different types of blocks by their origin, which might also
    obey different symmetries:

        - Self interaction blocks: These are blocks that encode the interactions between basis functions of the
          same point. These blocks are always square matrices. They are located at the diagonal of the matrix.
          If the matrix is symmetric, these blocks must also be symmetric.

        - Interaction blocks: All the rest of blocks, that contain interactions between basis functions from different
          points. Even if the matrix is symmetric, these blocks do not need to be symmetric. For each pair of points `ij`,
          there are two blocks: `ij` and `ji` However, if the matrix is symmetric, one block is the transpose of the other.
          Therefore, we only need to compute/predict one of them.

    **How it is implemented.**

    This module is implemented as a graph neural network. Block creating functions are convolved over
    edges and nodes to create the matrix blocks. Even though the matrix is computed with a convolution,
    we can not use a single function. There are two type of blocks that are different in nature:

        - Self interaction blocks: Convolved over nodes. Since each point type has a different basis,
          it will produce a block of different size. Therefore, we need **one function per point type**.

        - Interaction blocks: Convolved over edges. For the same reason as the self interaction blocks,
          we need one function per combination of point types.

    Each function is a `MatrixBlock` module.

    Parameters
    ----------
    unique_basis:
        A list with all the unique point basis (one for each point type) that this module
        should be able to handle. The inputs passed on forward do not necessarily need to
        contain all of them.
    irreps_in:
        Shorthand to set `irreps_in` for both `node_operation_kwargs` and `edge_operation_kwargs`.
        It will be ignored if the `*_operation_kwargs` argument already has an `"irreps_in"`
        key.
    node_operation:
        The operation used to compute the values for matrix blocks corresponding to
        self interactions (nodes).
        This is passed directly to the `MatrixBlock` class, see the `operation_cls`
        parameter there.
    node_operation_kwargs:
        Initialization arguments for the `node_operation` class.
        Same as `operation_kwargs` argument in `MatrixBlock`.
    edge_operation:
        The operation used to compute the values for matrix blocks corresponding to
        interactions between different nodes (edges).
        This is passed directly to the `MatrixBlock` class, see the `operation_cls`.
    edge_operation_kwargs:
        Initialization arguments for the `edge_operation` class.
        Same as `operation_kwargs` argument in `MatrixBlock`.
    node_irreps_in:
        The irreps that this module will accept for the node features. The order
        of the irreps needs to be at least as high as the maximum order found in
        the basis.
    edge_irreps_in:
        The irreps that this module will accept for the edge features. The order
        of the irreps needs to be at least as high as the maximum order found in
        the basis.
    symmetric:
        Whether the matrix is symmetric. If it is, edge blocks for edges connecting
        the same two atoms but in opposite directions will be computed only once (the
        block for the opposite direction is the transpose block).
    blocks_symmetry:
        The symmetry that each point block must obey. By default no symmetries are assumed.
    self_blocks_symmetry:
        The symmetry that node blocks must obey. If this is `None`:

          - If `symmetric` is `False`, self_blocks are assumed to have the same symmetry
            as other blocks, which is specified in the `blocks_symmetry` parameter.
          - If `symmetric` is `True`, self_blocks are assumed to be symmetric.
    """

    unique_basis: List[PointBasis]

    # List of self interaction functions (which compute node blocks).
    self_interactions: List[MatrixBlock]
    # Dictionary of interaction functions (which compute edge blocks).
    interactions: Dict[Tuple[int, int], MatrixBlock]

    def __init__(
        self,
        unique_basis: Sequence[PointBasis],
        preprocessing_nodes: Type = None,
        preprocessing_nodes_kwargs: dict = {},
        preprocessing_edges: Type = None,
        preprocessing_edges_kwargs: dict = {},
        node_operation: Type = None,
        node_operation_kwargs: dict = {},
        edge_operation: Type = None,
        edge_operation_kwargs: dict = {},
        symmetric: bool = False,
        blocks_symmetry: str = "ij",
        self_blocks_symmetry: Union[str, None] = None,
        matrix_block_cls: Type[MatrixBlock] = MatrixBlock,
        # numpy: ModuleType = np,
        self_interactions_list: Callable = list,
        interactions_dict: Callable = dict,
    ):
        super().__init__()

        # Determine the symmetry of self blocks if it is not provided.
        if self_blocks_symmetry is None:
            if symmetric:
                self_blocks_symmetry = "ij=ji"
            else:
                self_blocks_symmetry = blocks_symmetry

        self.symmetric = symmetric
        self.unique_basis = list(unique_basis)
        self._matrix_block_cls = matrix_block_cls
        self.numpy = numpy
        self._self_interactions_list = self_interactions_list
        self._interactions_dict = interactions_dict

        if preprocessing_nodes is None:
            self.preprocessing_nodes = None
        else:
            self.preprocessing_nodes = preprocessing_nodes(**preprocessing_nodes_kwargs)

        if preprocessing_edges is None:
            self.preprocessing_edges = None
        else:
            self.preprocessing_edges = preprocessing_edges(**preprocessing_edges_kwargs)

        # Build all the unique self-interaction functions (interactions of a point with itself)
        self_interactions = self._init_self_interactions(
            symmetry=self_blocks_symmetry,
            operation_cls=node_operation,
            preprocessor=self.preprocessing_nodes,
            **node_operation_kwargs,
        )
        self.self_interactions = self._self_interactions_list(self_interactions)
        # Do the same for interactions between different points (interactions of a point with its neighbors)
        interactions = self._init_interactions(
            symmetry=blocks_symmetry,
            operation_cls=edge_operation,
            preprocessor=self.preprocessing_edges,
            **edge_operation_kwargs,
        )
        self.interactions = self._interactions_dict(interactions)

    def _init_self_interactions(self, **kwargs) -> List[MatrixBlock]:
        self_interactions = []

        for point_type_basis in self.unique_basis:
            if len(point_type_basis.basis) == 0:
                # The point type has no basis functions
                self_interactions.append(None)
            else:
                self_interactions.append(
                    self._matrix_block_cls(
                        i_basis=point_type_basis,
                        j_basis=point_type_basis,
                        **kwargs,
                    )
                )

        return self_interactions

    def _init_interactions(self, **kwargs) -> Dict[Tuple[int, int], MatrixBlock]:
        point_type_combinations = itertools.combinations_with_replacement(
            range(len(self.unique_basis)), 2
        )

        interactions = {}

        for edge_type, (point_type, neigh_type) in enumerate(point_type_combinations):
            perms = [(edge_type, point_type, neigh_type)]

            # If the matrix is not symmetric, we need to include the opposite interaction
            # as well.
            if not self.symmetric and point_type != neigh_type:
                perms.append((-edge_type, neigh_type, point_type))

            for signed_edge_type, point_i, point_j in perms:
                i_basis = self.unique_basis[point_i]
                j_basis = self.unique_basis[point_j]

                if len(i_basis.basis) == 0 or len(j_basis.basis) == 0:
                    # One of the involved point types has no basis functions
                    interactions[point_i, point_j, signed_edge_type] = None
                else:
                    interactions[
                        point_i, point_j, signed_edge_type
                    ] = self._matrix_block_cls(
                        i_basis=i_basis,
                        j_basis=j_basis,
                        symm_transpose=neigh_type == point_type,
                        **kwargs,
                    )

        return {str(k): v for k, v in interactions.items()}

    def get_preprocessing_nodes_summary(self) -> str:
        """Returns a summary of the preprocessing nodes functions."""
        return str(self.preprocessing_nodes)

    def get_preprocessing_edges_summary(self) -> str:
        """Returns a summary of the preprocessing edges functions."""
        return str(self.preprocessing_edges)

    def get_node_operation_summary(self, node_operation: MatrixBlock) -> str:
        """Returns a summary of the node operation."""

        if hasattr(node_operation, "get_summary"):
            return node_operation.get_summary()
        else:
            try:
                return str(node_operation.operation.__class__.__name__)
            except AttributeError:
                return str(node_operation)

    def get_edge_operation_summary(self, edge_operation: MatrixBlock) -> str:
        """Returns a summary of the edge operation."""
        if hasattr(edge_operation, "get_summary"):
            return edge_operation.get_summary()
        else:
            try:
                return str(edge_operation.operation.__class__.__name__)
            except AttributeError:
                return str(edge_operation)

    @property
    def summary(self) -> str:
        """High level summary of the architecture of the module.

        It is better than the pytorch repr to understand the high level
        architecture of the module, but it is not as detailed.
        """

        s = ""

        s += f"Preprocessing nodes: {self.get_preprocessing_nodes_summary()}\n"

        s += f"Preprocessing edges: {self.get_preprocessing_edges_summary()}\n"

        s += "Node operations:"
        for i, x in enumerate(self.self_interactions):
            point = self.unique_basis[i]

            if x is None:
                s += f"\n ({point.type}) No basis functions."
                continue

            s += f"\n ({point.type}) "

            if x.symm_transpose:
                s += " [XY = YX.T]"

            s += f" {self.get_node_operation_summary(x)}"

        s += "\nEdge operations:"
        for k, x in self.interactions.items():
            point_type, neigh_type, edge_type = map(int, k[1:-1].split(","))

            point = self.unique_basis[point_type]
            neigh = self.unique_basis[neigh_type]

            if x is None:
                s += f"\n ({point.type}, {neigh.type}) No basis functions."
                continue

            s += f"\n ({point.type}, {neigh.type})"

            if x.symm_transpose:
                s += " [XY = YX.T]"

            s += f" {self.get_edge_operation_summary(x)}."

        return s

    def forward(
        self,
        data: BasisMatrixData,
        node_feats: ArrayType,
        preprocessing_nodes_kwargs: dict = {},
        preprocessing_edges_kwargs: dict = {},
        node_kwargs: Dict[str, ArrayType] = {},
        edge_kwargs: Dict[str, ArrayType] = {},
        global_kwargs: dict = {},
        node_operation_node_kwargs: Dict[str, ArrayType] = {},
        node_operation_global_kwargs: dict = {},
        edge_operation_node_kwargs: Dict[str, ArrayType] = {},
        edge_operation_global_kwargs: dict = {},
    ) -> Tuple[ArrayType, ArrayType]:
        """Computes the matrix elements.

        **VERY IMPORTANT NOTE**

        Edges are assumed to be sorted in a very specific way:

          - Opposite directions of the same edge should come consecutively.
          - The direction that has a positive edge type should come first. The "positive" direction
            in an edge {i, j}, between point types "type_i" and "type_j" is the direction from the
            smallest point type to the biggest point type.
          - Sorted by edge type within the same structure. That is, edges where the same two species interact should
            be grouped within each structure in the batch. These groups should be ordered by edge type.

        This is all taken care of by `BasisMatrixData`, so if you use it you don't need to worry about it.

        Parameters
        -----------
        data:
            The data object containing the graph information.
            It can also be a dictionary that mocks the `BasisMatrixData` object
            with the appropiate keys.
        node_kwargs: Dict[str, ArrayType] = {},
            Arguments to pass to node and edge operations that are node-wise.
            Tensors should have shape (n_nodes, ...).

            If you want to pass a node-wise argument only to node/edge operations,
            you should pass it on `{node/edge}_operation_node_kwargs`.

            The arguments passed here will be added to both `node_operation_node_kwargs` and
            `edge_operation_node_kwargs`. See those parameters for more information
            on how they are used.

            If a key is present in both `node_kwargs` and `*_operation_node_kwargs`,
            the value in `*_operation_node_kwargs` will be used.
        edge_kwargs: Dict[str, ArrayType] = {},
            Arguments to pass to edge operations that are edge-wise.
            Tensors should have shape (n_edges, ...).

            The module will filter and organize them to pass a tuple (type X, type -X) for
            edge operation X. That is, the tuple will contain both directions of the edge.

            NOTE: One can think of passing edge-wise arguments to the node operations, which
            can then be aggregated into node-wise arguments. However, all this module does with
            node-wise and endge-wise arguments is to organize and reshape them. Therefore, an
            aggregation operation should be done outside of this module.
        global_kwargs: dict = {},
            Arguments to pass to node and edge operations that are global (e.g. neither
            node-wise nor edge-wise). They are used by the operations as provided.
        node_operation_node_kwargs: Dict[str, ArrayType] = {}
            Arguments to pass to node operations that are node-wise.
            Tensors should have shape (n_nodes, ...).

            The module will filter them to contain only the values for nodes of type X
            before passing them to function for node type X.
        node_operation_global_kwargs: dict = {},
            Arguments to pass to node operations that are global. They will be passed
            to each function as provided.
        edge_operation_node_kwargs: Dict[str, ArrayType] = {},
            Arguments to pass to edge operations that are node-wise.
            Tensors should have shape (n_edges, ...).

            The module will filter and organize them to pass a tuple (type X, type Y) for
            edge operation X -> Y.
        edge_operation_global_kwargs: dict = {},
            Arguments to pass to edge operations that are global. They will be passed
            to each function as provided.

        Returns
        -----------
        node_labels:
            All the node blocks, flattened and concatenated.
        edge_blocks:
            All the edge blocks, flattened and concatenated.
        """

        # If there are preprocessing functions for the computation of nodes
        # or edges, apply them and overwrite the node_feats to be passed
        # to node/edge operations.
        # Note that preprocessing functions can return either a single value
        # (new_node_feats) or a tuple (new_node_feats, edge_messages).
        if self.preprocessing_nodes is not None:
            preprocessing_out = self.preprocessing_nodes(
                data=data, node_feats=node_feats, **preprocessing_nodes_kwargs
            )

            if isinstance(preprocessing_out, tuple):
                node_feats_for_nodes, edge_messages = preprocessing_out
            else:
                node_feats_for_nodes = preprocessing_out
                edge_messages = None

            if node_feats_for_nodes is not None:
                node_operation_node_kwargs = {
                    "node_feats": node_feats_for_nodes,
                    **node_operation_node_kwargs,
                }
        else:
            node_operation_node_kwargs = {
                "node_feats": node_feats,
                **node_operation_node_kwargs,
            }

        if self.preprocessing_edges is not None:
            preprocessing_out = self.preprocessing_edges(
                data=data, node_feats=node_feats, **preprocessing_edges_kwargs
            )

            if isinstance(preprocessing_out, tuple):
                node_feats_for_edges, edge_messages = preprocessing_out
            else:
                node_feats_for_edges = preprocessing_out
                edge_messages = None

            if node_feats_for_edges is not None:
                edge_operation_node_kwargs = {
                    "node_feats": node_feats_for_edges,
                    **edge_operation_node_kwargs,
                }
            if edge_messages is not None:
                edge_kwargs = {"edge_messages": edge_messages, **edge_kwargs}
        else:
            edge_operation_node_kwargs = {
                "node_feats": node_feats,
                **edge_operation_node_kwargs,
            }

        # Build the arguments to pass to each kind of operation (node/edge)
        node_operation_node_kwargs = {**node_kwargs, **node_operation_node_kwargs}
        edge_operation_node_kwargs = {**node_kwargs, **edge_operation_node_kwargs}

        node_operation_global_kwargs = {**global_kwargs, **node_operation_global_kwargs}
        edge_operation_global_kwargs = {**global_kwargs, **edge_operation_global_kwargs}

        # Compute node blocks using the self interaction functions.
        node_labels = self._forward_self_interactions(
            node_types=data["point_types"],
            node_kwargs=node_operation_node_kwargs,
            global_kwargs=node_operation_global_kwargs,
        )

        # Compute edge blocks using the interaction functions.
        edge_labels = self._forward_interactions(
            edge_types=data["edge_types"],
            edge_index=data["edge_index"],
            edge_type_nlabels=data["edge_type_nlabels"],
            node_kwargs=edge_operation_node_kwargs,
            edge_kwargs=edge_kwargs,
            global_kwargs=edge_operation_global_kwargs,
        )

        # Return both the node and edge labels.
        return (node_labels, edge_labels)

    def _forward_self_interactions(
        self,
        node_types: ArrayType,
        node_kwargs,
        global_kwargs,
    ) -> ArrayType:
        # Allocate a list where we will store the outputs of all node blocks.
        n_nodes = len(node_types)
        node_labels = [None] * n_nodes

        # Call each unique self interaction function with only the features
        # of nodes that correspond to that type.
        for node_type, func in enumerate(self.self_interactions):
            if func is None:
                continue

            # Select the features for nodes of this type
            mask = node_types == node_type

            # Quick exit if there are no features of this type
            if not mask.any():
                continue

            filtered_kwargs = {key: value[mask] for key, value in node_kwargs.items()}

            # If there are, compute the blocks.
            output = func(**filtered_kwargs, **global_kwargs)
            # Flatten the blocks
            output = output.reshape(output.shape[0], -1)

            for i, individual_output in zip(mask.nonzero(), output):
                node_labels[i] = individual_output

        return self.numpy.concatenate(
            [labels for labels in node_labels if labels is not None]
        )

    def _forward_interactions_init_arrays_kwargs(self, edge_types_array):
        return {}

    def _forward_interactions(
        self,
        edge_types: ArrayType,
        edge_index: ArrayType,
        edge_type_nlabels: ArrayType,
        node_kwargs: Dict[str, ArrayType] = {},
        edge_kwargs: Dict[str, ArrayType] = {},
        global_kwargs: dict = {},
    ):
        # THEN, COMPUTE EDGE BLOCKS
        # Allocate space to store all the edge labels.
        # Within the same structure, edges are grouped (and sorted) by edge type. The edge_type_nlabels tensor contains,
        # for each structure, the number of edges of each type. We can accumulate these values to get a pointer to the
        # beginning of each edge type in each structure.
        init_arrays_kwargs = self._forward_interactions_init_arrays_kwargs(edge_types)
        unique_edge_types = edge_type_nlabels.shape[1]
        edge_type_ptrs = self.numpy.zeros(
            edge_type_nlabels.shape[0] * edge_type_nlabels.shape[1] + 1,
            dtype=self.numpy.int64,
            **init_arrays_kwargs,
        )
        self.numpy.cumsum(edge_type_nlabels.ravel(), dim=0, out=edge_type_ptrs[1:])
        # Then we can allocate a tensor to store all of them.
        edge_labels = self.numpy.empty(
            edge_type_ptrs[-1],
            dtype=self.numpy.get_default_dtype(),
            **init_arrays_kwargs,
        )

        # Call each unique interaction function with only the features
        # of edges that correspond to that type.
        for module_key, func in self.interactions.items():
            if func is None:
                # Case where one of the point types has no basis functions.
                continue

            # The key of the module is the a tuple (int, int, int) converted to a string.
            point_type, neigh_type, edge_type = map(int, module_key[1:-1].split(","))

            # Get a mask to select the edges that belong to this type.
            mask = abs(edge_types) == abs(edge_type)
            if not mask.any():
                continue

            # Then, for all features, select only the edges of this type.
            filtered_edge_kwargs = {
                key: value[mask] for key, value in edge_kwargs.items()
            }
            type_edge_index = edge_index[:, mask]

            # Edges between the same points but in different directions are stored consecutively.
            # So we can select every 2 features to get the same direction for all edges.
            # For a block ij, we assume that the wanted direction is i -> j.
            # We always pass first the direction that the function is supposed to evaluate.
            if edge_type > 0:
                i_edges = slice(0, None, 2)
                j_edges = slice(1, None, 2)
            else:
                i_edges = slice(1, None, 2)
                j_edges = slice(0, None, 2)

            # Create the tuples of edge features. Each tuple contains the two directions of the
            # edge. The first item contains the "forward" direction, the second the "reverse" direction.
            filtered_edge_kwargs = {
                key: (value[i_edges], value[j_edges])
                for key, value in filtered_edge_kwargs.items()
            }

            # For the node arguments we need to filter them and create pairs, such that a tuple
            # (sender, receiver) is built for each node argument.
            filtered_node_kwargs = {
                key: (
                    value[type_edge_index[0, i_edges]],
                    value[type_edge_index[1, i_edges]],
                )
                for key, value in node_kwargs.items()
            }

            # Compute the outputs.
            # The output will be of shape [n_edges, i_basis_size, j_basis_size]. That is, one
            # matrix block per edge, where the shape of the block is determined by the edge type.
            output = func(
                **filtered_edge_kwargs, **filtered_node_kwargs, **global_kwargs
            )

            # Since each edge type has a different block shape, we need to flatten the blocks (and even
            # the n_edges dimension) to put them all in a single array.
            output = output.ravel()

            # Here, we fill the edge labels array with the output at the appropiate positions (as determined
            # when bulding the pointers before the compute loop).
            els = 0
            for start, end in zip(
                edge_type_ptrs[edge_type::unique_edge_types],
                edge_type_ptrs[edge_type + 1 :: unique_edge_types],
            ):
                next_els = els + end - start
                edge_labels[start:end] = output[els:next_els]
                els = next_els

        return edge_labels

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
