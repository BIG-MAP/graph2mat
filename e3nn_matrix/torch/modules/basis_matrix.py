"""File containing modules that compute matrices"""

from e3nn import o3
import itertools
import torch
from typing import Sequence, Type, Union, Tuple, List, Dict

from ...data.basis import PointBasis
from .node_readouts import NodeBlock, SimpleNodeBlock
from .edge_readouts import EdgeBlock, SimpleEdgeBlock


class MatrixBlock(torch.nn.Module):
    """Computes a fixed size matrix coming from the product of spherical harmonics.

    There are two things to note:
        - It computes a dense matrix.
        - It computes a fixed size matrix.

    It takes care of:
      - Determining what are the irreps needed to reproduce a certain block.
      - Converting from those irreps to the actual values of the block
        using the appropiate change of basis.

    This module doesn't implement any computation, so you need to pass one
    as stated in the ``operation`` parameter.

    Parameters
    -----------
    i_irreps: o3.Irreps
        The irreps of the matrix rows.
    j_irreps: o3.Irreps
        The irreps of the matrix columns.
    symmetry: str
        Symmetries that this matrix is expected to have. This should be indicated as documented
        in `e3nn.o3.ReducedTensorProducts`. As an example, for a symmetric matrix you would
        pass "ij=ji" here.
    operation_cls: Type[torch.nn.Module]
        Torch module used to actually do the computation. On initialization, it will receive
        the `irreps_out` argument from this module, specifying the shape of the output that
        it should produce.

        On forward, this module will just be a wrapper around the operation, so you should pass
        whatever arguments that the operation expects.
    **operation_kwargs: dict
        Any arguments needed for the initialization of the `operation_cls`.

    Returns
    -----------
    matrix: torch.Tensor
        A 2D tensor of shape (i_irreps.dim, j_irreps.dm) containing the output matrix.
    """

    block_shape: Tuple[int, int]
    block_size: int

    symm_transpose: bool

    _irreps_out: o3.Irreps

    def __init__(
        self,
        i_irreps: o3.Irreps,
        j_irreps: o3.Irreps,
        symmetry: str,
        operation_cls: Type[torch.nn.Module],
        symm_transpose: bool = False,
        **operation_kwargs,
    ):
        super().__init__()

        self.setup_reduced_tp(i_irreps=i_irreps, j_irreps=j_irreps, symmetry=symmetry)
        self.symm_transpose = symm_transpose

        self.operation = operation_cls(**operation_kwargs, irreps_out=self._irreps_out)

    def setup_reduced_tp(self, i_irreps: o3.Irreps, j_irreps: o3.Irreps, symmetry: str):
        # Store the shape of the block.
        self.block_shape = (i_irreps.dim, j_irreps.dim)
        # And number of elements in the block.
        self.block_size = i_irreps.dim * j_irreps.dim

        # Understand the irreps out that we need in order to create the block.
        # The block is a i_irreps.dim X j_irreps.dim matrix, with possible symmetries that can
        # reduce the number of degrees of freedom. We indicate this to the ReducedTensorProducts,
        # which we only use as a helper.
        reduced_tp = o3.ReducedTensorProducts(symmetry, i=i_irreps, j=j_irreps)
        self._irreps_out = reduced_tp.irreps_out

        # We also store the change of basis, a matrix that will bring us from the irreps_out
        # to the actual matrix block that we want to calculate.
        self.register_buffer("change_of_basis", reduced_tp.change_of_basis)

    def forward(self, *args, **kwargs):
        def compute_block(*args, **kwargs):
            # Get the irreducible output
            irreducible_out = self.operation(*args, **kwargs)

            # And convert it to the actual block of the matrix, using the change of basis
            # matrix stored on initialization.
            # n = number of nodes, i = dim of irreps, x = rows in block, y = cols in block
            return torch.einsum("ni,ixy->nxy", irreducible_out, self.change_of_basis)

        if self.symm_transpose == False:
            return compute_block(*args, **kwargs)
        else:
            forward = compute_block(*args, **kwargs)

            back_args = [
                (arg[1], arg[0]) if isinstance(arg, tuple) and len(arg) == 2 else arg
                for arg in args
            ]
            back_kwargs = {
                key: (value[1], value[0])
                if isinstance(value, tuple) and len(value) == 2
                else value
                for key, value in kwargs.items()
            }
            backward = compute_block(*back_args, **back_kwargs)

            return (forward + backward.transpose(-1, -2)) / 2


class BasisMatrixReadout(torch.nn.Module):
    """Computes a variable size sparse matrix coming from the products of spherical harmonics.

    ## Design concept

    The module builds the matrix block by block. We define a block as a region of the matrix
    where the rows are all the basis of a given point, and all the columns are the basis of another
    given point. There are then two clearly different types of blocks by their origin, which might also
    obey different symmetries:
      -  Self interaction blocks: These are blocks that encode the interactions between basis functions of the
      same point. These blocks are always square matrices. They are located at the diagonal of the matrix.
      If the matrix is symmetric, these blocks must also be symmetric.
      -  Interaction blocks: All the rest of blocks, that contain interactions between basis functions from different
      points. Even if the matrix is symmetric, these blocks do not need to be symmetric. For each pair of points `ij`,
      there are two blocks: `ij` and `ji` However, if the matrix is symmetric, one block is the transpose of the other.
      Therefore, we only need to compute/predict one of them.

    ## How it is implemented.

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
    unique_basis: Sequence[PointBasis]
        A list with all the unique point basis (one for each point type) that this module
        should be able to handle. The inputs passed on forward do not necessarily need to
        contain all of them.
    node_operation: Type[torch.nn.Module]
        The operation used to compute the values for matrix blocks corresponding to
        self interactions (nodes).
        This is passed directly to the `MatrixBlock` class, see the `operation_cls`
        parameter there.
    node_operation_kwargs: dict
        Initialization arguments for the `node_operation` class.
        Same as `operation_kwargs` argument in `MatrixBlock`.
    edge_operation: Type[torch.nn.Module]
        The operation used to compute the values for matrix blocks corresponding to
        interactions between different nodes (edges).
        This is passed directly to the `MatrixBlock` class, see the `operation_cls`.
    edge_operation_kwargs: dict
        Initialization arguments for the `edge_operation` class.
        Same as `operation_kwargs` argument in `MatrixBlock`.
    node_irreps_in: o3.Irreps
        The irreps that this module will accept for the node features. The order
        of the irreps needs to be at least as high as the maximum order found in
        the basis.
    edge_irreps_in: o3.Irreps
        The irreps that this module will accept for the edge features. The order
        of the irreps needs to be at least as high as the maximum order found in
        the basis.
    symmetric: bool, optional
        Whether the matrix is symmetric. If it is, edge blocks for edges connecting
        the same two atoms but in opposite directions will be computed only once (the
        block for the opposite direction is the transpose block).
    blocks_symmetry: str, optional
        The symmetry that each point block must obey. By default no symmetries are assumed.
    self_blocks_symmetry: str, optional
        The symmetry that node blocks must obey. If this is `None`:
          - If `symmetric` is `False`, self_blocks are assumed to have the same symmetry
            as other blocks, which is specified in the `blocks_symmetry` parameter.
          - If `symmetric` is `True`, self_blocks are assumed to be symmetric.
    """

    def __init__(
        self,
        unique_basis: Sequence[PointBasis],
        node_operation: Type[NodeBlock] = SimpleNodeBlock,
        node_operation_kwargs: dict = {},
        edge_operation: Type[EdgeBlock] = SimpleEdgeBlock,
        edge_operation_kwargs: dict = {},
        symmetric: bool = False,
        blocks_symmetry: str = "ij",
        self_blocks_symmetry: Union[str, None] = None,
    ):
        super().__init__()

        # Determine the symmetry of self blocks if it is not provided.
        if self_blocks_symmetry is None:
            if symmetric:
                self_blocks_symmetry = "ij=ji"
            else:
                self_blocks_symmetry = blocks_symmetry

        self.symmetric = symmetric

        # Find out the basis irreps for each unique type of point
        self._unique_basis = unique_basis
        self._basis_irreps = [point_basis.irreps for point_basis in unique_basis]

        # Build all the unique self-interaction functions (interactions of a point with itself)
        self_interactions = self._init_self_interactions(
            basis_irreps=self._basis_irreps,
            symmetry=self_blocks_symmetry,
            operation_cls=node_operation,
            **node_operation_kwargs,
        )
        # And store them in a module list
        self.self_interactions = torch.nn.ModuleList(self_interactions)

        # Do the same for interactions between different points (interactions of a point with its neighbors)
        interactions = self._init_interactions(
            basis_irreps=self._basis_irreps,
            symmetry=blocks_symmetry,
            operation_cls=edge_operation,
            **edge_operation_kwargs,
        )
        self.interactions = torch.nn.ModuleDict(
            {str(k): v for k, v in interactions.items()}
        )

    def _init_self_interactions(self, basis_irreps, **kwargs) -> List[torch.nn.Module]:
        self_interactions = []

        for point_type_irreps in basis_irreps:
            self_interactions.append(
                MatrixBlock(
                    i_irreps=point_type_irreps,
                    j_irreps=point_type_irreps,
                    **kwargs,
                )
            )

        return self_interactions

    def _init_interactions(
        self, basis_irreps, **kwargs
    ) -> Tuple[Dict[Tuple[int, int], torch.nn.Module]]:
        point_type_combinations = itertools.combinations_with_replacement(
            range(len(basis_irreps)), 2
        )

        interactions = {}

        for edge_type, (point_type, neigh_type) in enumerate(point_type_combinations):
            perms = [(edge_type, point_type, neigh_type)]

            # If the matrix is not symmetric, we need to include the opposite interaction
            # as well.
            if not self.symmetric and point_type != neigh_type:
                perms.append((-edge_type, neigh_type, point_type))

            for signed_edge_type, point_i, point_j in perms:
                interactions[point_i, point_j, signed_edge_type] = MatrixBlock(
                    i_irreps=basis_irreps[point_i],
                    j_irreps=basis_irreps[point_j],
                    symm_transpose=neigh_type == point_type,
                    **kwargs,
                )

        return interactions

    @property
    def summary(self) -> str:
        """High level summary of the architecture of the module.

        It is better than the pytorch repr to understand the high level
        architecture of the module, but it is not as detailed.
        """

        s = ""

        s += "Node operations:"
        for i, x in enumerate(self.self_interactions):
            point = self._unique_basis[i]
            s = (
                s
                + f"\n ({point.type}) {str(x.operation.__class__.__name__)}: ({point.irreps})^2 -> {x._irreps_out}"
            )

            if x.symm_transpose:
                s += " [XY = YX.T]"

        s += "\nEdge operations:"
        for k, x in self.interactions.items():
            point_type, neigh_type, edge_type = map(int, k[1:-1].split(","))

            point = self._unique_basis[point_type]
            neigh = self._unique_basis[neigh_type]

            s = (
                s
                + f"\n ({point.type}, {neigh.type}) {str(x.operation.__class__.__name__)}: ({point.irreps}) x ({neigh.irreps}) -> {x._irreps_out}."
            )

            if x.symm_transpose:
                s += " [XY = YX.T]"

        return s

    def forward(
        self,
        node_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_type_nlabels: torch.Tensor,
        node_operation_node_kwargs: Dict[str, torch.Tensor] = {},
        node_operation_global_kwargs: dict = {},
        edge_operation_node_kwargs: Dict[str, torch.Tensor] = {},
        edge_operation_edge_kwargs: Dict[str, torch.Tensor] = {},
        edge_operation_global_kwargs: dict = {},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the matrix.

        VERY IMPORTANT: Edges are assumed to be sorted in a very specific way:
          - Opposite directions of the same edge should come consecutively.
          - The direction that has a positive edge type should come first. The "positive" direction
            in an edge {i, j}, between point types "type_i" and "type_j" is the direction from the
            smallest point type to the biggest point type.
          - Sorted by edge type within the same structure. That is, edges where the same two species interact should
            be grouped within each structure in the batch. These groups should be ordered by edge type.

        Parameters
        -----------
        node_types: torch.Tensor
            Array of length (n_nodes, ) containing the type for each node.
        edge_index: torch.Tensor
            Array of shape (2, n_edges) storing, for each edge, the pair of point indices
            that are connected through it.
        edge_types: torch.Tensor
            Array of length (edge_feats, ) containing the type for each node.
        edge_type_nlabels: torch.Tensor
            Helper array to understand how to store edge labels after they are computed.
            This array should be (len(batch), n_edge_types). It should contain, for each
            structure in the batch, the amount of matrix elements that correspond to
            blocks of a certain edge type. This amount comes from multiplying the number
            of edges of this type by the size of the matrix block that they need.
        node_feats, node_attrs, edge_feats, edge_attrs: torch.Tensor
            tensors coming from the model that this function is reading out from.
            Node tensors should be (n_nodes, X) and edge tensors (n_edges, X), where X
            is the dimension of the irreps specified on initialization for each of these
            tensors.
        node_operation_node_kwargs: Dict[str, torch.Tensor] = {}
            Arguments to pass to node operations that are node-wise.
            The module will filter them to contain only the values for nodes of type X
            before passing them to function for node type X.
        node_operation_global_kwargs: dict = {},
            Arguments to pass to node operations that are global. They will be passed
            to each function as provided.
        edge_operation_node_kwargs: Dict[str, torch.Tensor] = {},
            Arguments to pass to edge operations that are node-wise.
            The module will filter and organize them to pass a tuple (type X, type Y) for
            edge operation X -> Y.
        edge_operation_edge_kwargs: Dict[str, torch.Tensor] = {},
            Arguments to pass to edge operations that are edge-wise.
            The module will filter and organize them to pass a tuple (type X, type -X) for
            edge operation X. That is, the tuple will contain both directions of the edge.
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
        # Compute node blocks using the self interaction functions.
        node_labels = self._forward_self_interactions(
            node_types=node_types,
            node_kwargs=node_operation_node_kwargs,
            global_kwargs=node_operation_global_kwargs,
        )

        # Compute edge blocks using the interaction functions.
        edge_labels = self._forward_interactions(
            edge_types=edge_types,
            edge_index=edge_index,
            edge_type_nlabels=edge_type_nlabels,
            node_kwargs=edge_operation_node_kwargs,
            edge_kwargs=edge_operation_edge_kwargs,
            global_kwargs=edge_operation_global_kwargs,
        )

        # Return both the node and edge labels.
        return (node_labels, edge_labels)

    def _forward_self_interactions(
        self, node_types: torch.Tensor, node_kwargs, global_kwargs
    ) -> torch.Tensor:
        # Allocate a list where we will store the outputs of all node blocks.
        n_nodes = len(node_types)
        node_labels = [None] * n_nodes

        # Call each unique self interaction function with only the features
        # of nodes that correspond to that type.
        for node_type, func in enumerate(self.self_interactions):
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

        return torch.concatenate(node_labels)

    def _forward_interactions(
        self,
        edge_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_nlabels: torch.Tensor,
        node_kwargs: Dict[str, torch.Tensor] = {},
        edge_kwargs: Dict[str, torch.Tensor] = {},
        global_kwargs: dict = {},
    ):
        # THEN, COMPUTE EDGE BLOCKS
        # Allocate space to store all the edge labels.
        # Within the same structure, edges are grouped (and sorted) by edge type. The edge_type_nlabels tensor contains,
        # for each structure, the number of edges of each type. We can accumulate these values to get a pointer to the
        # beginning of each edge type in each structure.
        unique_edge_types = edge_type_nlabels.shape[1]
        edge_type_ptrs = torch.zeros(
            edge_type_nlabels.shape[0] * edge_type_nlabels.shape[1] + 1,
            dtype=torch.int64,
            device=edge_types.device,
        )
        torch.cumsum(edge_type_nlabels.ravel(), dim=0, out=edge_type_ptrs[1:])
        # Then we can allocate a tensor to store all of them.
        edge_labels = torch.empty(
            edge_type_ptrs[-1],
            dtype=torch.get_default_dtype(),
            device=edge_types.device,
        )

        # Call each unique interaction function with only the features
        # of edges that correspond to that type.
        for module_key, func in self.interactions.items():
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
