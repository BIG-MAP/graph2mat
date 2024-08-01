from e3nn import o3
import torch

from typing import Sequence, Type, Union, Optional, Dict, Literal

from graph2mat import PointBasis, MatrixBlock

from graph2mat.bindings.torch import TorchGraph2Mat

from .matrixblock import E3nnIrrepsMatrixBlock
from .node_operations import E3nnSimpleNodeBlock
from .edge_operations import E3nnSimpleEdgeBlock

__all__ = ["E3nnGraph2Mat"]


class E3nnGraph2Mat(TorchGraph2Mat):
    """Extension of `TorchGraph2Mat` to deal with irreps.

    Parameters
    ----------
    unique_basis:
        Basis of the point types that the function should be able to handle.
        It can either be a list of the unique `PointBasis` objects
        or a `BasisTableWithEdges` object.

        Note that when using the function, each graph does not need to contain
        all the point types.
    irreps:
        Dictionary containing the irreps of all the possible features that the model
        has to deal with.

        The only required key is: `"node_feats_irreps"`.

        The rest depend on what the preprocessing and block producing functions
        use.
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
    **kwargs:
        Additional arguments passed to the `Graph2Mat` class.

    Examples
    --------

    This is an example of how to use it with custom node and edge operations,
    which will allow you to understand what the operation receives so that
    you can tune it to your needs:

    .. code-block:: python

        import torch
        from e3nn import o3

        from graph2mat import PointBasis
        from graph2mat.bindings.e3nn import E3nnGraph2Mat

        # Build a basis set
        basis = [
            PointBasis("A", R=2, basis=[1], basis_convention="cartesian"),
            PointBasis("B", R=5, basis=[2, 1], basis_convention="cartesian")
        ]

        # Define the custom operation that just prints the arguments
        class CustomOperation(torch.nn.Module):

            def __init__(self, node_feats_irreps, irreps_out):
                print("INITIALIZING OPERATION")
                print("INPUT NODE FEATS IRREPS:", node_feats_irreps)
                print("IRREPS_OUT:", irreps_out)
                print("")

            def __call__(self, node_feats):
                print(data, node_feats)

                # This return will create an error. Instead, you should
                # produce something of irreps_out.
                return node_feats

        # Initialize the module
        g2m = E3nnGraph2Mat(
            unique_basis=basis,
            irreps={"node_feats_irreps": o3.Irreps("2x0e + 1x1o")},
            symmetric=True,
            node_operation=CustomOperation,
            edge_operation=CustomOperation,
        )

        print("SUMMARY")
        print(g2m.summary)

    See Also
    --------
    Graph2Mat
        The class that `E3nnGraph2Mat` extends. Its documentation contains a more
        detailed explanation of the inner workings of the class.

    """

    def __init__(
        self,
        unique_basis: Sequence[PointBasis],
        irreps: Dict[str, o3.Irreps],
        preprocessing_nodes: Optional[Type[torch.nn.Module]] = None,
        preprocessing_nodes_kwargs: dict = {},
        preprocessing_edges: Optional[Type[torch.nn.Module]] = None,
        preprocessing_edges_kwargs: dict = {},
        preprocessing_edges_reuse_nodes: bool = True,
        node_operation: Type = E3nnSimpleNodeBlock,
        node_operation_kwargs: dict = {},
        edge_operation: Type = E3nnSimpleEdgeBlock,
        edge_operation_kwargs: dict = {},
        symmetric: bool = False,
        blocks_symmetry: str = "ij",
        self_blocks_symmetry: Union[str, None] = None,
        matrix_block_cls: Type[MatrixBlock] = E3nnIrrepsMatrixBlock,
        **kwargs,
    ):
        self.irreps = irreps

        preprocessing_nodes_kwargs = {"irreps": irreps, **preprocessing_nodes_kwargs}
        preprocessing_edges_kwargs = {"irreps": irreps, **preprocessing_edges_kwargs}

        node_operation_kwargs = {"irreps": irreps, **node_operation_kwargs}
        edge_operation_kwargs = {"irreps": irreps, **edge_operation_kwargs}

        super().__init__(
            unique_basis=unique_basis,
            preprocessing_nodes=preprocessing_nodes,
            preprocessing_nodes_kwargs=preprocessing_nodes_kwargs,
            preprocessing_edges=preprocessing_edges,
            preprocessing_edges_kwargs=preprocessing_edges_kwargs,
            preprocessing_edges_reuse_nodes=preprocessing_edges_reuse_nodes,
            node_operation=node_operation,
            node_operation_kwargs=node_operation_kwargs,
            edge_operation=edge_operation,
            edge_operation_kwargs=edge_operation_kwargs,
            matrix_block_cls=matrix_block_cls,
            symmetric=symmetric,
            blocks_symmetry=blocks_symmetry,
            self_blocks_symmetry=self_blocks_symmetry,
            **kwargs,
        )

    def _get_readout_irreps(
        self,
        preprocessor,
        irreps: Dict[str, o3.Irreps],
        what: Literal["nodes", "edges"],
    ) -> Dict[str, o3.Irreps]:
        """Possibly updates the irreps if there is a preprocessing step."""
        if preprocessor is not None:
            irreps_out = preprocessor.irreps_out

            if isinstance(irreps_out, o3.Irreps):
                node_feats_irreps = irreps_out
                edge_message_irreps = None
            else:
                # Otherwise it's a tuple with node and edge message irreps
                node_feats_irreps, edge_message_irreps = irreps_out

            # If the edge preprocessor doesn't return nodes irreps, we reuse the nodes irreps
            # from the input in case the reuse of node features is enabled.
            if (
                node_feats_irreps is None
                and what == "edges"
                and self.preprocessing_edges_reuse_nodes
            ):
                node_feats_irreps = irreps["node_feats_irreps"]

            irreps = {
                **irreps,
                "node_feats_irreps": node_feats_irreps,
                "edge_messages_irreps": edge_message_irreps,
            }

        return irreps

    def _init_self_interactions(
        self, *args, preprocessor=None, irreps: Dict[str, o3.Irreps] = {}, **kwargs
    ):
        readout_irreps = self._get_readout_irreps(preprocessor, irreps, "nodes")

        return super()._init_self_interactions(
            *args, **kwargs, preprocessor=preprocessor, irreps=readout_irreps
        )

    def _init_interactions(
        self, *args, preprocessor=None, irreps: Dict[str, o3.Irreps] = {}, **kwargs
    ):
        readout_irreps = self._get_readout_irreps(preprocessor, irreps, "edges")

        return super()._init_interactions(
            *args, **kwargs, preprocessor=preprocessor, irreps=readout_irreps
        )
