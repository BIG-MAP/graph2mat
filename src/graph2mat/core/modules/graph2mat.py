"""Graph2Mat, the models' skeleton."""

import itertools
import numpy as np
from typing import (
    Sequence,
    Type,
    Union,
    Tuple,
    List,
    Dict,
    TypeVar,
    Generic,
    Callable,
    Optional,
)
from types import ModuleType

from ..data import BasisMatrixData, BasisTableWithEdges
from .matrixblock import MatrixBlock
from ..data.basis import PointBasis

__all__ = ["Graph2Mat"]

# This type will be
ArrayType = TypeVar("ArrayType")


class Graph2Mat(Generic[ArrayType]):
    r"""Converts a graph to a sparse matrix.

    The matrix that this module computes has variable size, which corresponds
    to the size of the graph. It is built by applying a convolution of its functions
    over the edges and nodes of the graph.

    **High level architecture overview**

    .. image:: /_static/images/Graph2Mat.svg

    `Graph2mat` builds the matrix

    .. math:: M_{\nu\mu}

    block by block. We define a block as a region of the matrix where the rows are all the basis of a given point,
    and all the columns are the basis of another given point. I.e. given two points :math:`(i, j)`:

    .. math:: M_{ij} = all \space M_{\nu\mu} \space \text{where} \space \nu \in i,  \mu \in j

    The shape of the basis of points :math:`i` and :math:`j` determines then the shape of the block :math:`M_{ij}`.
    Therefore, we need a different function to produce each kind of block. There are two clearly different
    types of blocks by their origin, which might also obey different symmetries, and therefore we can classify
    the blocks in two categories:

        - **Self interaction blocks** (:math:`f_n`): These are blocks that encode the interactions between basis functions of the
          same point. They correspond to nodes in the graph. These blocks are always square matrices. They are located at the diagonal of the matrix.
          If the matrix is symmetric, these blocks must also be symmetric.

        - **Interaction blocks** (:math:`f_e`): All the rest of blocks that contain interactions between basis functions from different
          points. They correspond to edges in the graph. Even if the matrix is symmetric, these blocks do not need to be symmetric.
          For each pair of points :math:`(i, j)`, there are two blocks: :math:`M_{ij}` and :math:`M_{ji}`.
          However, if the matrix is symmetric, one block is the transpose of the other. Therefore in that case
          we only need to compute/predict one of them.

    Node features from the graph are passed to the block producing functions. Each block producing function
    only receives the features that correspond to the blocks that it needs to produce, as depicted in the
    sketch.

    Optionally, one can pass preprocessing functions :math:`(p_n, p_e)` that update the graph before passing
    it to the node/edge block producing functions. The edge preprocessing function can also return edge-wise
    messages.

    .. note::

        `Graph2Mat` itself is not a learnable module. If you are doing machine learning, the only
        learnable parameters will be in the node/edge operations :math:`f` and  the preprocessing
        functions :math:`p` functions.

        `Graph2Mat` is just a skeleton so that you can quickly experiment with
        different functions.

    .. warning::

        It is very likely that you need an extension like ``TorchGraph2Mat`` or ``E3nnGraph2Mat``
        in practice to do machine learning, as those set the appropriate defaults
        and add some extra things that are particular for the frameworks.

    Parameters
    ----------
    unique_basis:
        Basis of the point types that the function should be able to handle.
        It can either be a list of the unique `PointBasis` objects
        or a `BasisTableWithEdges` object.

        Note that when using the function, each graph does not need to contain
        all the point types.
    preprocessing_nodes:
        A module that preprocesses the node features before passing them to the
        node block producing functions. This is :math:`p_n` in the sketch.

        It should be a class with an `__init__` method that receives the initialization
        arguments and a `__call__` method that receives the data to process. The data
        will be the same that has been passed to `Graph2Mat`.

        It can output either a single array (the updated node features) or a tuple
        (updated node features, edge messages). In the second case, edge messages
        will be disregarded, this is just so that the preprocessing functions can be
        reused for nodes and edge processing.
    preprocessing_nodes_kwargs:
        Initialization arguments passed directly to the `preprocessing_nodes` class.
    preprocessing_edges:
        A module that preprocesses the edge features before passing them to the
        edge block producing functions. This is :math:`p_e` in the sketch.

        It should be a class with an `__init__` method that receives the initialization
        arguments and a `__call__` method that receives the data to process. The data
        will be the same that has been passed to `Graph2Mat`.

        It can output either a single array (the updated node features) or a tuple
        (updated node features, edge messages). In the second case, the updated node
        features can be `None`.
    preprocessing_edges_kwargs:
        Initialization arguments passed directly to the `preprocessing_edges` class.
    preprocessing_edges_reuse_nodes:
        If there is a preprocessing function for edges and it only returns edge messages,
        whether the un-updated node features should also be passed to the edge block producing
        functions.

        It has no effect if there is no edge preprocessing function or the edge preprocessing
        function returns both node features and edge messages.
    node_operation:
        The operation used to compute the values for matrix blocks corresponding to
        self interactions (nodes). This is the :math:`f_n` functions in the sketch.

        It should be a class with an `__init__` method that receives the initialization
        arguments (such as `i_basis`, `j_basis` and `symmetry`) and a `__call__` method that
        receives the data to process. It will receive the node features for the node blocks
        that the operation must compute.
    node_operation_kwargs:
        Initialization arguments for the `node_operation` class.
    edge_operation:
        The operation used to compute the values for matrix blocks corresponding to
        interactions between different nodes (edges). This is the :math:`f_e` functions
        in the sketch.

        It should be a class with an `__init__` method that receives the initialization
        arguments (such as `i_basis`, `j_basis` and `symmetry`) and a `__call__` method that
        receives the data to process. It will receive:

        - Node features as a tuple: (feats_senders, feats_receiver)
        - Edge messages as a tuple: (edge_message_ij, edge_message_ji)

        Each item in the tuples is an array with length `n_edges`.

        The operation does not need to handle permutation of the nodes. If the matrix is symmetric,
        permutation of nodes should lead to the transposed block, but this is handled by `Graph2Mat`.
    edge_operation_kwargs:
        Initialization arguments for the `edge_operation` class.
    symmetric:
        Whether the matrix is symmetric. If it is, edge blocks for edges connecting
        the same two atoms but in opposite directions will be computed only once (the
        block for the opposite direction is the transpose block).

        This also determines the `symmetry` argument pass to the `node_operation`
        on initialization.
    blocks_symmetry:
        The symmetry that each block (both edge and node blocks) must obey. If
        the blocks must be symmetric for example, this should be set to `"ij=ji"`.
    self_blocks_symmetry:
        The symmetry that node blocks must obey. If this is `None`:

          - If `symmetric` is `False`, self_blocks are assumed to have the same symmetry
            as other blocks, which is specified in the `blocks_symmetry` parameter.
          - If `symmetric` is `True`, self_blocks are assumed to be symmetric.
    matrix_block_cls:
        Class that wraps matrix block operations.
    numpy:
        Module used as `numpy`. This can for example be set to `torch`. If `None`, we use
        `numpy`.
    self_interactions_list:
        Wrapper for the list of self interaction functions (:math:`f_n`, node blocks).

        This is for example used in `torch` to convert the list of functions to a `torch.nn.ModuleList`.
    interactions_dict:
        Wrapper for the dictionary of interaction functions (:math:`f_e`, edge blocks).

        This is for example used in `torch` to convert the dictionary of functions to a `torch.nn.ModuleDict`.

    Examples
    ---------

    This is an example of how to use it with custom node and edge operations,
    which will allow you to understand what the operation receives so that
    you can tune it to your needs:

    .. code-block:: python

        from graph2mat import Graph2Mat, PointBasis

        # Build a basis set
        basis = [
            PointBasis("A", R=2, basis=[1], basis_convention="cartesian"),
            PointBasis("B", R=5, basis=[2, 1], basis_convention="cartesian")
        ]

        # Define the custom operation that just prints the arguments
        class CustomOperation:

            def __init__(self, i_basis, j_basis, symmetry):
                print("INITIALIZING OPERATION")
                print("I_BASIS", i_basis)
                print("J_BASIS", j_basis)
                print("SYMMETRY", symmetry)
                print()

            def __call__(self, **kwargs):
                print(kwargs)
                return kwargs

        # Initialize the module
        g2m = Graph2Mat(
            unique_basis=basis,
            symmetric=True,
            node_operation=CustomOperation,
            edge_operation=CustomOperation,
        )

        print("SUMMARY")
        print(g2m.summary)

    """
    #: The table holding all information about the basis. This is an internal
    #: table created by the module from `unique_basis`, but it should probably
    #: be equal to the basis table that you use to process your data.
    basis_table: BasisTableWithEdges

    #: List of self interaction functions (which compute node blocks).
    self_interactions: List[MatrixBlock]
    #: Dictionary of interaction functions (which compute edge blocks).
    interactions: Dict[Tuple[int, int], MatrixBlock]

    def __init__(
        self,
        unique_basis: Union[BasisTableWithEdges, Sequence[PointBasis]],
        preprocessing_nodes: Optional[Type] = None,
        preprocessing_nodes_kwargs: dict = {},
        preprocessing_edges: Optional[Type] = None,
        preprocessing_edges_kwargs: dict = {},
        preprocessing_edges_reuse_nodes: bool = True,
        node_operation: Type = None,
        node_operation_kwargs: dict = {},
        edge_operation: Type = None,
        edge_operation_kwargs: dict = {},
        symmetric: bool = False,
        blocks_symmetry: str = "ij",
        self_blocks_symmetry: Union[str, None] = None,
        matrix_block_cls: Type[MatrixBlock] = MatrixBlock,
        numpy: Optional[ModuleType] = None,
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
        self.basis_table = (
            unique_basis
            if isinstance(unique_basis, BasisTableWithEdges)
            else BasisTableWithEdges(unique_basis)
        )
        self._matrix_block_cls = matrix_block_cls
        self.numpy = numpy if numpy is not None else np
        self._self_interactions_list = self_interactions_list
        self.node_operation_cls = node_operation
        self._interactions_dict = interactions_dict
        self.edge_operation_cls = edge_operation

        if preprocessing_nodes is None:
            self.preprocessing_nodes = None
        else:
            self.preprocessing_nodes = preprocessing_nodes(**preprocessing_nodes_kwargs)

        self.preprocessing_edges_reuse_nodes = preprocessing_edges_reuse_nodes
        if preprocessing_edges is None:
            self.preprocessing_edges = None
        else:
            self.preprocessing_edges = preprocessing_edges(**preprocessing_edges_kwargs)

        assert (
            node_operation is not None
        ), f"{self.__class__.__name__} needs a node operation to be provided."
        assert (
            edge_operation is not None
        ), f"{self.__class__.__name__} needs an edge operation to be provided."

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

        for point_type_basis in self.basis_table.basis:
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
            range(len(self.basis_table.basis)), 2
        )

        interactions = {}

        for edge_type, (point_type, neigh_type) in enumerate(point_type_combinations):
            perms = [(edge_type, point_type, neigh_type)]

            # If the matrix is not symmetric, we need to include the opposite interaction
            # as well.
            if not self.symmetric and point_type != neigh_type:
                perms.append((-edge_type, neigh_type, point_type))

            for signed_edge_type, point_i, point_j in perms:
                i_basis = self.basis_table.basis[point_i]
                j_basis = self.basis_table.basis[point_j]

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

    def _get_preprocessing_nodes_summary(self) -> str:
        """Returns a summary of the preprocessing nodes functions."""
        return str(self.preprocessing_nodes)

    def _get_preprocessing_edges_summary(self) -> str:
        """Returns a summary of the preprocessing edges functions."""
        return str(self.preprocessing_edges)

    def _get_node_operation_summary(self, node_operation: MatrixBlock) -> str:
        """Returns a summary of the node operation."""

        if hasattr(node_operation, "get_summary"):
            return node_operation.get_summary()
        else:
            try:
                return str(node_operation.operation.__class__.__name__)
            except AttributeError:
                return str(node_operation)

    def _get_edge_operation_summary(self, edge_operation: MatrixBlock) -> str:
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

        s += f"Preprocessing nodes: {self._get_preprocessing_nodes_summary()}\n"

        s += f"Preprocessing edges: {self._get_preprocessing_edges_summary()}\n"

        s += "Node operations:"
        for i, x in enumerate(self.self_interactions):
            point = self.basis_table.basis[i]

            if x is None:
                s += f"\n ({point.type}) No basis functions."
                continue

            s += f"\n ({point.type}) "

            if x.symm_transpose:
                s += " [XY = YX.T]"

            s += f" {self._get_node_operation_summary(x)}"

        s += "\nEdge operations:"
        for k, x in self.interactions.items():
            point_type, neigh_type, edge_type = map(int, k[1:-1].split(","))

            point = self.basis_table.basis[point_type]
            neigh = self.basis_table.basis[neigh_type]

            if x is None:
                s += f"\n ({point.type}, {neigh.type}) No basis functions."
                continue

            s += f"\n ({point.type}, {neigh.type})"

            if x.symm_transpose:
                s += " [XY = YX.T]"

            s += f" {self._get_edge_operation_summary(x)}."

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

        .. note::

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

            if node_feats_for_edges is None and self.preprocessing_edges_reuse_nodes:
                node_feats_for_edges = node_feats

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
            data=data,
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
        data: BasisMatrixData,
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

        edge_kwargs = {**edge_kwargs}
        edge_requirements = getattr(self.edge_operation_cls, "_data_get_edge_args", ())
        for key in edge_requirements:
            edge_kwargs[key] = data[key]

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
                **filtered_node_kwargs, **filtered_edge_kwargs, **global_kwargs
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

    def _forward_interactions_with_resorting(
        self,
        edge_types: ArrayType,
        edge_index: ArrayType,
        edge_type_nlabels: ArrayType,
        node_kwargs: Dict[str, ArrayType] = {},
        edge_kwargs: Dict[str, ArrayType] = {},
        global_kwargs: dict = {},
    ):
        """Experimental interactions forward by building a resorting array with cython. (NOT USED FOR NOW)

        The resorting indices don't need gradients, so we can just compute them with cython.

        For the tests that I have done it doesn't show a speedup (i.e. the resorting is not a bottleneck)
        """

        outputs = []

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

            outputs.append(output)

        unsorted_edge_labels = self.numpy.concatenate(outputs)

        sort_indices = self._get_interactions_resort_index(edge_types)

        return unsorted_edge_labels[sort_indices]

    def _get_interactions_resort_index(self, edge_types, **kwargs):
        """Experimental resorting array building with cython. NOT USED FOR NOW"""
        from ._labels_resort import get_edgelabels_resorting_array

        edge_types = edge_types[::2].numpy(force=True)

        sizes = self.basis_table.edge_block_size

        indices = get_edgelabels_resorting_array(edge_types.astype(int), sizes)

        indices = self.numpy.from_numpy(indices, **kwargs)
        return indices
