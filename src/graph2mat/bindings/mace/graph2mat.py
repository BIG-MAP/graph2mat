from e3nn import o3
import torch
from typing import Sequence, Type, Union, Tuple, Optional, Dict

from mace.modules.blocks import (
    RadialEmbeddingBlock,
)

from graph2mat import PointBasis

from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.bindings.e3nn import (
    E3nnGraph2Mat,
    E3nnSimpleEdgeBlock,
    E3nnSimpleNodeBlock,
    E3nnInteraction,
    E3nnEdgeMessageBlock,
)

# from ..node_readouts import NodeBlock, SimpleNodeBlock
# from ..edge_readouts import EdgeBlock, SimpleEdgeBlock
from .preprocessing import MACEInteraction, MACEEdgeMessageBlock

# from .edge_readouts import MACEEdgeBlock

__all__ = [
    "MACEGraph2Mat",
    "MACEStandaloneGraph2Mat",
]

# wewe

# TODO: Create an E3nnIdentityInteraction that does nothing.
# Make E3nn interactions have a property that indicates the irreps
# of their output (and also whether they output both nodes and edge messages)
# irreps_out could be a tuple:
# preprocessing.irreps_out = (node_feats_irreps, edge_message_irreps)
# Then, try to remove the need for passing the node/edge_operation irreps,
# which is awful.
# After that, try to incorporate the MACEEdgeBlockNodeMix!
# For that, it probably makes sense to automatically parse the BasisMatrixTorchData
# so that we can filter it to pass the edge features to the edge readout blocks.
# This would mean that we can use "data" also in the readout blocks, and probably
# we can get rid of MACEGraph2Mat altogether!
# Second TODO: Add try: except: blocks on Graph2Mat when calling the functions, if
# some function is missing an argument, the error should indicate how to pass it
# through the Graph2Mat interface.
# Third TODO: Check that we can store and load the models, and that they work with the
# server!
# Fourth TODO: Make E3nn interactions possibly standalone. E.g. that they can compute
# the edge embeddings.
# Fifth TODO: Recreate documentation and publish!


class MACEGraph2Mat(E3nnGraph2Mat):
    """Basis matrix readout that uses MACE message passing.

    It relies on some quantities being precomputed and therefore it is meant
    to be integrated into MACE models.
    """

    # Whether the edge block is specific to MACE or not.
    _MACE_edge_operation: bool

    def __init__(
        self,
        irreps: Dict[str, o3.Irreps],
        avg_num_neighbors: float,
        unique_basis: Sequence[PointBasis],
        preprocessing_nodes: Optional[Type[torch.nn.Module]] = E3nnInteraction,
        preprocessing_edges: Optional[Type[torch.nn.Module]] = E3nnEdgeMessageBlock,
        node_operation: Type[torch.nn.Module] = E3nnSimpleNodeBlock,
        edge_operation: Type[torch.nn.Module] = E3nnSimpleEdgeBlock,
        symmetric: bool = False,
        blocks_symmetry: str = "ij",
        self_blocks_symmetry: Union[str, None] = None,
        **kwargs,
    ):
        super().__init__(
            unique_basis=unique_basis,
            irreps=irreps,
            preprocessing_nodes=preprocessing_nodes,
            preprocessing_nodes_kwargs={
                "avg_num_neighbors": avg_num_neighbors,
            },
            preprocessing_edges=preprocessing_edges,
            node_operation=node_operation,
            edge_operation=edge_operation,
            symmetric=symmetric,
            blocks_symmetry=blocks_symmetry,
            self_blocks_symmetry=self_blocks_symmetry,
            **kwargs,
        )

    def forward(
        self,
        data: TorchBasisMatrixData,
        node_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the matrix.

        See BasisMatrixReadout.forward for more details.

        Parameters
        -----------
        data:
            The data object containing the graph information.
            It can also be a dictionary that mocks the `BasisMatrixData` object
            with the appropiate keys.
        node_feats, edge_feats, edge_attrs: torch.Tensor
            tensors coming from the model that this function is reading out from.
            Node tensors should be (n_nodes, X) and edge tensors (n_edges, X), where X
            is the dimension of the irreps specified on initialization for each of these
            tensors.
        Returns
        -----------
        node_labels:
            All the node blocks, flattened and concatenated.
        edge_blocks:
            All the edge blocks, flattened and concatenated.
        """

        edge_operation_node_kwargs = {}
        edge_operation_edge_kwargs = {}
        if False:  # self._MACE_edge_operation:
            edge_operation_node_kwargs = {"node_feats": node_feats}
            edge_operation_edge_kwargs = {
                "edge_feats": data["edge_feats"],
                "edge_index": data["edge_index"].transpose(0, 1),
            }

        return super().forward(
            data=data,
            node_feats=node_feats,
            edge_operation_node_kwargs=edge_operation_node_kwargs,
            edge_kwargs=edge_operation_edge_kwargs,
        )


class MACEStandaloneGraph2Mat(MACEGraph2Mat):
    """Basis matrix readout that uses MACE message passing and incorporates edge embeddings.

    It is a MACE-like readout, but it doesn't necessarily need to be used within a MACE
    model. Inside it, it contains the embeddings for the edge direction and length that are
    used in MACE, so that the readout can be used by passing raw edge vectors and lengths.
    """

    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        edge_hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        unique_basis: Sequence[PointBasis],
        node_operation: Type[torch.nn.Module] = E3nnSimpleNodeBlock,
        edge_operation: Type[torch.nn.Module] = E3nnSimpleEdgeBlock,
        symmetric: bool = False,
        blocks_symmetry: str = "ij",
        self_blocks_symmetry: Union[str, None] = None,
        interaction_cls: Type[torch.nn.Module] = MACEInteraction,
        edge_msg_cls: Type[torch.nn.Module] = MACEEdgeMessageBlock,
    ):
        # Node attrs is a simple one hot encoding of the atomic numbers.
        node_attrs_irreps = o3.Irreps([(len(unique_basis), (0, 1))])

        # Radial function for edges, which will compute edge_feats.
        radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{radial_embedding.out_dim}x0e")

        # Spherical harmonics that encode the direction of the edges.
        # This will compute edge_attrs.
        edge_attrs_irreps = o3.Irreps.spherical_harmonics(max_ell)
        spherical_harmonics = o3.SphericalHarmonics(
            edge_attrs_irreps, normalize=True, normalization="component"
        )

        # Initialize the MAECBasisMatrixReadout.
        super().__init__(
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            edge_hidden_irreps=edge_hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            unique_basis=unique_basis,
            node_operation=node_operation,
            edge_operation=edge_operation,
            symmetric=symmetric,
            blocks_symmetry=blocks_symmetry,
            self_blocks_symmetry=self_blocks_symmetry,
            interaction_cls=interaction_cls,
            edge_msg_cls=edge_msg_cls,
        )

        # Store the modules
        self.radial_embedding = radial_embedding
        self.spherical_harmonics = spherical_harmonics

    def forward(
        self,
        data: TorchBasisMatrixData,
        node_feats: torch.Tensor,  # [n_nodes, node_feats_irreps.dim]
        edge_vectors: torch.Tensor,  # [n_edges, 3]
        edge_lengths: torch.Tensor,  # [n_edges, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the matrix.

        See BasisMatrixReadout.forward for more details.

        Parameters
        -----------
        data:
            The data object containing the graph information.
            It can also be a dictionary that mocks the `BasisMatrixData` object
            with the appropiate keys.
        node_feats:
            Node features, shape (n_nodes, X), where X is the dimension of the node
            features irreps.

        Returns
        -----------
        node_labels:
            All the node blocks, flattened and concatenated.
        edge_blocks:
            All the edge blocks, flattened and concatenated.
        """
        edge_attrs = self.spherical_harmonics(edge_vectors)
        edge_feats = self.radial_embedding(edge_lengths)

        return super().forward(
            data=data,
            node_feats=node_feats,
            edge_feats=edge_feats,
            edge_attrs=edge_attrs,
        )
