"""File containing all modules needed to create  basis-basis matrices from graphs.
The main module are OrbitalMatrixReadout and its subclasses, which accept all the graph.
"""

from e3nn import o3
import itertools
import torch
from typing import Sequence, Type, Union, Tuple

from ...data.basis import PointBasis
from .node_readouts import NodeBlock, SimpleNodeBlock
from .edge_readouts import EdgeBlock, SimpleEdgeBlock
from .messages import EdgeMessageBlock, NodeMessageBlock


class MatrixBlock(torch.nn.Module):
    """Module that computes a block of a matrix with known symmetries for its rows and columns.
    As an example, this can be used to compute the blocks of a density matrix (DM)
    in DFT calculations, where rows and columns are basis orbitals that come from 
    spherical harmonics.
    It takes care of:
      - Determining what are the irreps needed to reproduce a certain block.
      - Converting from those irreps to the actual values of the block
        using the appropiate change of basis.
    This module doesn't implement any computation, so you need to pass one
    as stated in the ``operation`` parameter.
    Parameters
    -----------
    feats_irreps: o3.Irreps
        The irreps of the features that the module will accept when computing.
    i_irreps: o3.Irreps
        The irreps of the rows of the block that this module will generate.
    j_irreps: o3.Irreps
        The irreps of the columns of the block that this module will generate.
    block_symmetry: str
        Symmetries that this block is expected to have. This should be indicated as documented
        in `e3nn.o3.ReducedTensorProducts`. As an example, for a symmetric block you would
        pass "ij=ji" here.
    operation: class that inherits from torch.nn.Module, optional
        Torch module that will receive as arguments the input irreps (irreps_in: o3.Irreps) 
        that should accept and the output irreps (irreps_out: o3.Irreps) that it should produce.
        On evaluation mode, it will receive the corresponding features, with dimension as
        stated by irreps_in. This operation therefore does not need to know anything about
        the DM or the block that it is computing, it's just a module that blindly does the
        work. 
    """

    def __init__(self,
        i_irreps: o3.Irreps, j_irreps: o3.Irreps, block_symmetry: str,
        operation: Type[torch.nn.Module], 
        symm_transpose: bool = False,
        **operation_kwargs
    ):
        super().__init__()

        self.setup_reduced_tp(i_irreps=i_irreps, j_irreps=j_irreps, block_symmetry=block_symmetry)

        # If the operation supports transpose symmetric blocks f(x, y) == f(y, x).T, ask for it.
        # Otherwise, ask for the normal operation and we might need to symmetrize the block later
        # if symm_transpose is True.
        try: 
            self.operation = operation(symm_transpose=symm_transpose, **operation_kwargs, irreps_out=self._irreps_out)

            self.symm_transpose = False
        except TypeError:
            self.operation = operation(**operation_kwargs, irreps_out=self._irreps_out)

            self.symm_transpose = symm_transpose

    def setup_reduced_tp(self, i_irreps: o3.Irreps, j_irreps: o3.Irreps, block_symmetry: str):
        # Store the shape of the block.
        self.block_shape = (i_irreps.dim, j_irreps.dim)
        # And number of elements in the block.
        self.block_size = i_irreps.dim * j_irreps.dim

        # Understand the irreps out that we need in order to create the block. 
        # The block is a i_irreps.dim X j_irreps.dim matrix, with possible symmetries that can
        # reduce the number of degrees of freedom. We indicate this to the ReducedTensorProducts, 
        # which we only use as a helper.
        reduced_tp = o3.ReducedTensorProducts(block_symmetry, i=i_irreps, j=j_irreps)
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

            back_args = [(arg[1], arg[0]) if isinstance(arg, tuple) and len(arg) == 2 else arg for arg in args]
            back_kwargs = {key: (value[1], value[0]) if isinstance(value, tuple) and len(value) == 2 else value for key, value in kwargs.items()}
            backward = compute_block(*back_args, **back_kwargs)

            return (forward + backward.transpose(-1, -2)) / 2

class BasisMatrixReadout(torch.nn.Module):
    """Module responsible for generating an basis-basis matrix from a graph.

    ## Basis-basis matrix
    
    We refer to basis-basis matrix in general to any matrix whose both rows
    and columns are basis functions centered around a point.
   
    ## Building it block by block 

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
    The row and column size of each block are defined by the basis size of each point.
    Therefore, this module needs to create multiple functions.
      - For self interaction blocks: One function per point type is required.
      - For interaction blocks: One function per combination of point types is required.

    Parameters
    ----------
    node_irreps_in: o3.Irreps
        The irreps that this module will accept for the node features. The order 
        of the irreps needs to be at least as high as the maximum order found in 
        the basis.
    edge_irreps_in: o3.Irreps
        The irreps that this module will accept for the edge features. The order 
        of the irreps needs to be at least as high as the maximum order found in 
        the basis.
    unique_basis:
        A list with all the unique point basis that this module should be able to handle.
    node_operation: class that inherits from torch.nn.Module
        The operation used to compute the values to build DM blocks corresponding to
        self interactions (nodes).
        This is passed directly to the `MatrixBlock` class, see the `operation` parameter
        there.
    edge_operation: class that inherits from torch.nn.Module
        The operation used to compute the values to build DM blocks corresponding to
        interactions between different nodes (edges).
        This is passed directly to the `MatrixBlock` class, see the `operation` parameter
        there.
    symmetric: bool, optional
        Whether the matrix is symmetric. If it is, edges for the same bond but different directions
        will be reduced into one.
    blocks_symmetry: str, optional
        The symmetry that each point block must obey. By default no symmetries are assumed.
    self_blocks_symmetry: str, optional
        The symmetry that blocks containing interactions within the same point must obey. If
        this is `None`:
          - If `symmetric` is `False`, self_blocks are assumed to have the same symmetry 
            as other blocks, which is specified in the `blocks_symmetry` parameter.
          - If `symmetric` is `True`, self_blocks are assumed to be symmetric.
    """

    def __init__(self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        edge_hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        unique_basis: Sequence[PointBasis],
        node_operation: Type[NodeBlock] = SimpleNodeBlock, edge_operation: Type[EdgeBlock] = SimpleEdgeBlock,
        symmetric: bool = False,
        blocks_symmetry: str = "ij", self_blocks_symmetry: Union[str, None] = None,
        interaction_cls: Type[torch.nn.Module] = NodeMessageBlock,
        edge_msg_cls: Type[torch.nn.Module] = EdgeMessageBlock,
    ):
        super().__init__()

        # Determine the symmetry of self blocks if it is not provided.
        if self_blocks_symmetry is None:
            if symmetric:
                self_blocks_symmetry = "ij=ji"
            else:
                self_blocks_symmetry = blocks_symmetry

        self.symmetric = symmetric

        # Initialize the lists to store all unique functions that we need.
        self.self_interactions = torch.nn.ModuleList([])
        self.interactions = torch.nn.ModuleList([])
        self.interactions_edge_type = []

        # Find out the basis irreps for each unique type of point
        self._basis_irreps = [point_basis.irreps for point_basis in unique_basis]

        # Function that will compute the messages for each node.
        self.node_messages = interaction_cls(
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=node_feats_irreps,
            hidden_irreps=node_feats_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )

        # Now build all the unique functions for self interactions
        for point_type_irreps in self._basis_irreps:
            self.self_interactions.append(
                MatrixBlock(
                    i_irreps=point_type_irreps,
                    j_irreps=point_type_irreps,
                    block_symmetry=self_blocks_symmetry,
                    operation=node_operation,
                    irreps_in=node_feats_irreps,
                )
            )

        self.edge_messages = edge_msg_cls(
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=edge_hidden_irreps,
        )

        # And for all possible pairs of interactions.
        point_type_combinations = itertools.combinations_with_replacement(range(len(unique_basis)), 2)
        for edge_type, (point_type, neigh_type) in enumerate(point_type_combinations):

            perms = [(edge_type, point_type, neigh_type)]

            # If the matrix is not symmetric, we need to include the opposite interaction
            # as well.
            if not self.symmetric and point_type != neigh_type:
                perms.append((-edge_type, neigh_type, point_type))

            for signed_edge_type, point_i, point_j in perms:
                self.interactions_edge_type.append(signed_edge_type)

                self.interactions.append(
                    MatrixBlock(
                        i_irreps=self._basis_irreps[point_i],
                        j_irreps=self._basis_irreps[point_j],
                        block_symmetry=blocks_symmetry,
                        operation=edge_operation,
                        # Input parameters coming from the graph
                        edge_feats_irreps=edge_feats_irreps,
                        edge_messages_irreps=edge_hidden_irreps,
                        node_feats_irreps=node_feats_irreps,
                        symm_transpose=neigh_type == point_type,
                    )
                )

    def forward(self, 
        node_feats: torch.Tensor, node_attrs:torch.Tensor, 
        edge_feats: torch.Tensor, edge_attrs: torch.Tensor,
        node_types: torch.Tensor, edge_types: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_type_nlabels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the matrix.
        VERY IMPORTANT: All edge_* tensors are assumed to be sorted in a very specific
        way:
          - Opposite directions of the same edge should come consecutively.
          - The direction that has a positive edge type should come first. The "positive" direction
            in an edge {i, j}, between point types "type_i" and "type_j" is the direction from the
            smallest point type to the biggest point type.
          - Sorted by edge type within the same structure. That is, edges where the same two species interact should
            be grouped within each structure in the batch. These groups should be ordered by edge type.
        Parameters
        -----------
        node_feats, node_attrs, edge_feats, edge_attrs: torch.Tensor
            tensors coming from the model that this function is reading out from.
            Node tensors should be (n_nodes, X) and edge tensors (n_edges, X), where X
            is the dimension of the irreps specified on initialization for each of these
            tensors.
        node_types: torch.Tensor
            Array of length (n_nodes, ) containing the type for each node.
        edge_types: torch.Tensor
            Array of length (edge_feats, ) containing the type for each node.
        edge_index: torch.Tensor
            Array of shape (2, n_edges) storing, for each edge, the pair of point indices
            that are connected through it.
        edge_type_nlabels: torch.Tensor
            Helper array to understand how to store edge labels after they are computed.
            This array should be (len(batch), n_edge_types). It should contain, for each
            structure in the batch, the amount of matrix elements that correspond to
            blocks of a certain edge type. This amount comes from multiplying the number
            of edges of this type by the size of the matrix block that they need.
        Returns
        -----------
        node_labels:
            All the node blocks, flattened and concatenated.
        edge_blocks:
            All the edge blocks, flattened and concatenated.
        """

        # FIRST, COMPUTE NODE BLOCKS
        # Allocate a list where we will store the outputs of all node blocks.
        n_nodes = len(node_feats)
        node_labels = [None]*n_nodes

        node_messages = self.node_messages(
            node_attrs=node_attrs,
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )

        # Initialize variables that will used in the loop, so that we can delete
        # them safely when the loop ends.   
        type_messages = type_feats = mask = output = individual_output = None

        # Call each unique self interaction function with only the features 
        # of nodes that correspond to that type.
        for node_type, func in enumerate(self.self_interactions):

            # Select the features for nodes of this type
            mask = node_types == node_type
            type_feats = node_feats[mask]
            type_messages = node_messages[mask]

            # Quick exit if there are no features of this type
            if len(type_feats) == 0:
                continue

            # If there are, compute the blocks.
            output = func(node_feats=type_feats, node_messages=type_messages)
            # Flatten the blocks
            output = output.reshape(output.shape[0], -1)

            for i, individual_output in zip(mask.nonzero(), output):
                node_labels[i] = individual_output
        
        # Delete variables that are no longer needed to help reduce memory usage.
        del node_messages, type_messages, type_feats, output, mask, individual_output

        node_labels = torch.concatenate(node_labels)

        edge_messages = self.edge_messages(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )

        # THEN, COMPUTE EDGE BLOCKS
        # Allocate space to store all the edge labels.
        # Within the same structure, edges are grouped (and sorted) by edge type. The edge_type_nlabels tensor contains, 
        # for each structure, the number of edges of each type. We can accumulate these values to get a pointer to the 
        # beginning of each edge type in each structure. 
        unique_edge_types = edge_type_nlabels.shape[1]
        edge_type_ptrs = torch.zeros(edge_type_nlabels.shape[0] * edge_type_nlabels.shape[1] + 1, dtype=torch.int64, device=edge_feats.device)
        torch.cumsum(edge_type_nlabels.ravel(), dim=0, out=edge_type_ptrs[1:])
        # Then we can allocate a tensor to store all of them.
        edge_labels = torch.empty(edge_type_ptrs[-1], dtype=torch.get_default_dtype(), device=edge_feats.device)

        # Initialize variables that will used in the loop, so that we can delete
        # them safely when the loop ends.
        type_messages = type_feats = mask = output = None

        # Call each unique interaction function with only the features 
        # of edges that correspond to that type.
        for edge_type, func in zip(self.interactions_edge_type, self.interactions):

            # Get a mask to select the edges that belong to this type.
            mask = abs(edge_types) == edge_type

            # Then, for all features, select only the edges of this type.
            type_feats = edge_feats[mask]
            type_messages = edge_messages[mask]
            type_edge_index = edge_index[:, mask]

            # Quick exit if there are no edges of this type
            if len(type_feats) == 0:
                continue

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

            # Select the node features that correspond to the direction of the bond
            i_node_feats = node_feats[type_edge_index[0, i_edges]]
            j_node_feats = node_feats[type_edge_index[1, j_edges]]

            # Compute the outputs.
            # The output will be of shape [n_edges, i_basis_size, j_basis_size]. That is, one
            # matrix block per edge, where the shape of the block is determined by the edge type.
            output = func(
                edge_feats=(type_feats[i_edges], type_feats[j_edges]),
                edge_messages=(type_messages[i_edges], type_messages[j_edges]),
                edge_index=(type_edge_index[:, i_edges], type_edge_index[:, j_edges]),
                node_feats=(i_node_feats, j_node_feats)
            )

            # Since each edge type has a different block shape, we need to flatten the blocks (and even
            # the n_edges dimension) to put them all in a single array.
            output = output.ravel()

            # Here, we fill the edge labels array with the output at the appropiate positions (as determined
            # when bulding the pointers before the compute loop).
            els = 0
            for start, end in zip(edge_type_ptrs[edge_type::unique_edge_types], edge_type_ptrs[edge_type + 1::unique_edge_types]):
                next_els = els + end - start
                edge_labels[start:end] = output[els:next_els]
                els = next_els

        # Delete variables that are no longer needed to help reduce memory usage.
        del edge_messages, type_messages, type_feats, output, mask, edge_type_ptrs

        # Return both the node and edge labels.
        return (node_labels, edge_labels)

