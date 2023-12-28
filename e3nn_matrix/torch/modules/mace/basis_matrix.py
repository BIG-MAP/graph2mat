from e3nn import o3
import torch
from typing import Sequence, Type, Union, Tuple

from mace.modules.blocks import (
    RadialEmbeddingBlock,
)

from e3nn_matrix.data.basis import PointBasis

from ..basis_matrix import BasisMatrixReadout
from ..node_readouts import NodeBlock, SimpleNodeBlock
from ..edge_readouts import EdgeBlock, SimpleEdgeBlock
from .messages import MACEEdgeMessageBlock, MACENodeMessageBlock
from .edge_readouts import MACEEdgeBlock

__all__ = [
    "MACEBasisMatrixReadout",
    "StandaloneMACEBasisMatrixReadout",
]


class MACEBasisMatrixReadout(BasisMatrixReadout):
    """Basis matrix readout that uses MACE message passing.

    It relies on some quantities being precomputed and therefore it is meant
    to be integrated into MACE models.
    """

    # Whether the edge block is specific to MACE or not.
    _MACE_edge_operation: bool

    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        edge_hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        unique_basis: Sequence[PointBasis],
        node_operation: Type[NodeBlock] = SimpleNodeBlock,
        edge_operation: Type[EdgeBlock] = SimpleEdgeBlock,
        symmetric: bool = False,
        blocks_symmetry: str = "ij",
        self_blocks_symmetry: Union[str, None] = None,
        interaction_cls: Type[torch.nn.Module] = MACENodeMessageBlock,
        edge_msg_cls: Type[torch.nn.Module] = MACEEdgeMessageBlock,
    ):
        # Node and edge messages will be computed before the readout as a
        # preprocessing step to give more richness to the features.
        node_messages = interaction_cls(
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=node_feats_irreps,
            hidden_irreps=node_feats_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )

        edge_messages = edge_msg_cls(
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=edge_hidden_irreps,
        )

        # Then simply initialize the readout flow.

        # If the edge block is specific to MACE, it will understand MACE specific arguments.
        # Otherwise just pass the generic irreps_in argument.
        self._MACE_edge_operation = issubclass(edge_operation, MACEEdgeBlock)
        if self._MACE_edge_operation:
            edge_operation_kwargs = {
                "edge_feats_irreps": edge_feats_irreps,
                "edge_messages_irreps": edge_hidden_irreps,
                "node_feats_irreps": node_feats_irreps,
            }
        else:
            edge_operation_kwargs = {
                "irreps_in": edge_hidden_irreps,
            }

        super().__init__(
            unique_basis=unique_basis,
            node_operation=node_operation,
            node_operation_kwargs={
                "irreps_in": [node_feats_irreps, node_feats_irreps],
            },
            edge_operation=edge_operation,
            edge_operation_kwargs=edge_operation_kwargs,
            symmetric=symmetric,
            blocks_symmetry=blocks_symmetry,
            self_blocks_symmetry=self_blocks_symmetry,
        )

        # Store the modules
        self.node_messages = node_messages
        self.edge_messages = edge_messages

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        node_types: torch.Tensor,
        edge_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_nlabels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the matrix.

        See BasisMatrixReadout.forward for more details.

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
        node_messages = self.node_messages(
            node_attrs=node_attrs,
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )

        edge_messages = self.edge_messages(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )

        if self._MACE_edge_operation:
            edge_operation_node_kwargs = {"node_feats": node_feats}
            edge_operation_edge_kwargs = {
                "edge_feats": edge_feats,
                "edge_messages": edge_messages,
                "edge_index": edge_index.transpose(0, 1),
            }
        else:
            edge_operation_node_kwargs = {}
            edge_operation_edge_kwargs = {
                "edge_messages": edge_messages,
            }

        return super().forward(
            node_types=node_types,
            edge_index=edge_index,
            edge_types=edge_types,
            edge_type_nlabels=edge_type_nlabels,
            node_operation_node_kwargs={
                "node_feats": node_feats,
                "node_messages": node_messages,
            },
            node_operation_global_kwargs={},
            edge_operation_node_kwargs=edge_operation_node_kwargs,
            edge_kwargs=edge_operation_edge_kwargs,
            edge_operation_global_kwargs={},
        )


class StandaloneMACEBasisMatrixReadout(MACEBasisMatrixReadout):
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
        node_operation: Type[NodeBlock] = SimpleNodeBlock,
        edge_operation: Type[EdgeBlock] = SimpleEdgeBlock,
        symmetric: bool = False,
        blocks_symmetry: str = "ij",
        self_blocks_symmetry: Union[str, None] = None,
        interaction_cls: Type[torch.nn.Module] = MACENodeMessageBlock,
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
        node_feats: torch.Tensor,  # [n_nodes, node_feats_irreps.dim]
        node_attrs: torch.Tensor,  # [n_nodes, n_node_types] ]
        node_types: torch.Tensor,  # [n_nodes, ]
        edge_types: torch.Tensor,  # [n_edges, ]
        edge_index: torch.Tensor,  # [2, n_edges]
        edge_vectors: torch.Tensor,  # [n_edges, 3]
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        edge_type_nlabels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the matrix.

        See BasisMatrixReadout.forward for more details.

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
        edge_attrs = self.spherical_harmonics(edge_vectors)
        edge_feats = self.radial_embedding(edge_lengths)

        return super().forward(
            node_feats=node_feats,
            node_attrs=node_attrs,
            edge_feats=edge_feats,
            edge_attrs=edge_attrs,
            node_types=node_types,
            edge_types=edge_types,
            edge_index=edge_index,
            edge_type_nlabels=edge_type_nlabels,
        )
