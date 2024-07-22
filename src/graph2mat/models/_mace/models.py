"""Variant of the MACE model using the orbital matrix readouts."""

from typing import Any, Dict, Type, Sequence, Optional

import torch
from e3nn import o3

from mace.modules.blocks import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
)
from mace.modules.utils import get_edge_vectors_and_lengths

from graph2mat import PointBasis
from graph2mat.bindings.torch.data import BasisMatrixTorchData


class OrbitalMatrixMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        edge_hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        correlation: int,
        unique_basis: Sequence[PointBasis],
        matrix_readout: Type, 
        symmetric_matrix: bool,
        node_block_readout: Type[NodeBlock],
        edge_block_readout: Type[EdgeBlock],
        only_last_readout: bool,
        node_attr_irreps: Optional[o3.Irreps] = None,
    ):
        super().__init__()
        self.r_max = r_max
        self.num_elements = num_elements
        # Embedding
        if node_attr_irreps is None:
            node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(ir.ir), ir.ir) for ir in node_attr_irreps])

        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readout
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.only_last_readout = only_last_readout
        self.readouts = torch.nn.ModuleList()

        if not self.only_last_readout:
            self.readouts.append(
                matrix_readout(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    edge_hidden_irreps=edge_hidden_irreps,
                    avg_num_neighbors=avg_num_neighbors,
                    unique_basis=unique_basis,
                    symmetric=symmetric_matrix,
                    node_operation=node_block_readout,
                    edge_operation=edge_block_readout,
                )
            )

        for i in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)

            if i == num_interactions - 2 or not self.only_last_readout:
                readout = matrix_readout(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    edge_hidden_irreps=edge_hidden_irreps,
                    avg_num_neighbors=avg_num_neighbors,
                    unique_basis=unique_basis,
                    symmetric=symmetric_matrix,
                    node_operation=node_block_readout,
                    edge_operation=edge_block_readout,
                )

                self.readouts.append(readout)

    def forward(
        self,
        data: BasisMatrixTorchData,
        training=False,
    ) -> Dict[str, Any]:
        # Setup
        # This is only if we want to compute matrix gradients. For now, we don't.
        # data.positions.requires_grad = True

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_labels_contributions = []
        edge_labels_contributions = []

        if self.only_last_readout:
            readouts = [None] * (len(self.interactions) - 1)
            readouts.append(self.readouts[0])
        else:
            readouts = self.readouts

        for interaction, product, readout in zip(
            self.interactions, self.products, readouts
        ):
            # Do the message passing.
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )

            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"][:, :self.num_elements]
            )

            if readout is None:
                continue

            # Compute matrices. The matrix for each graph in the batch has potentially a
            # different size. Therefore, the return from readout will just be a
            # flat (1D) tensor. This means that we lose the edge/node dimension. This is not
            # a big problem though since we don't actually need to reduce all nodes/edges to compute a
            # graph property. I.e. in "normal" MACE we need to sum over the nodes of a graph
            # to get the total energy. In this case we just need to compare element by element
            # from predictions to target.
            inter_node_labels, inter_edge_labels = readout(
                node_feats=node_feats,
                node_attrs=data["node_attrs"],
                node_types=data.point_types,
                edge_feats=edge_feats,
                edge_attrs=edge_attrs,
                edge_types=data.edge_types,
                edge_index=data.edge_index,
                edge_type_nlabels=data.edge_type_nlabels,
            )

            node_labels_contributions.append(inter_node_labels)
            edge_labels_contributions.append(inter_edge_labels)

        # Sum over the matrix contributions of each message passing iteration.
        node_labels_contributions = torch.stack(node_labels_contributions, dim=-1)
        node_labels = torch.sum(node_labels_contributions, dim=-1)

        edge_labels_contributions = torch.stack(edge_labels_contributions, dim=-1)
        edge_labels = torch.sum(edge_labels_contributions, dim=-1)

        return {
            "node_labels": node_labels,
            "node_labels_contributions": node_labels_contributions,
            "edge_labels": edge_labels,
            "edge_labels_contributions": edge_labels_contributions,
        }
