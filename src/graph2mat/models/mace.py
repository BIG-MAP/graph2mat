import torch
from e3nn import o3
from copy import copy

from mace.modules import MACE

from graph2mat.bindings.e3nn import E3nnGraph2Mat

import torch
from e3nn import o3

from mace.modules import MACE

from mace.modules.utils import (
    get_edge_vectors_and_lengths,
)


class MatrixMACE(torch.nn.Module):
    """Model that wraps a MACE model to produce a matrix output."""

    def __init__(self, mace: MACE, readout_per_interaction: bool = False, **kwargs):
        super().__init__()

        self.mace = mace

        self.readout_per_interaction = readout_per_interaction

        edge_hidden_irreps = kwargs.pop("edge_hidden_irreps", None)

        if self.readout_per_interaction:
            self.mace_inter_irreps = [
                o3.Irreps(inter.hidden_irreps) for inter in self.mace.interactions
            ]

            self.matrix_readouts = torch.nn.ModuleList(
                [
                    E3nnGraph2Mat(
                        irreps=dict(
                            node_attrs_irreps=inter.node_attrs_irreps,
                            node_feats_irreps=o3.Irreps(inter.hidden_irreps),
                            edge_attrs_irreps=inter.edge_attrs_irreps,
                            edge_feats_irreps=inter.edge_feats_irreps,
                            edge_hidden_irreps=edge_hidden_irreps,
                        ),
                        **kwargs,
                    )
                    for inter in self.mace.interactions
                ]
            )
        else:
            self.mace_inter_irreps = sum(
                [inter.hidden_irreps for inter in self.mace.interactions], o3.Irreps()
            )

            self.matrix_readouts = E3nnGraph2Mat(
                irreps=dict(
                    node_attrs_irreps=self.mace.interactions[0].node_attrs_irreps,
                    node_feats_irreps=self.mace_inter_irreps,
                    edge_attrs_irreps=self.mace.interactions[0].edge_attrs_irreps,
                    edge_feats_irreps=self.mace.interactions[0].edge_feats_irreps,
                    edge_hidden_irreps=edge_hidden_irreps,
                ),
                **kwargs,
            )

    def forward(self, data, compute_force=False, **kwargs):
        mace_out = self.mace(data, compute_force=compute_force, **kwargs)

        # Compute edge feats and edge attrs from the modules in the mace model
        # (we can't access them from the model because they are not stored/outputted,
        # but they are very cheap to recompute)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.mace.spherical_harmonics(vectors)
        edge_feats = self.mace.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.mace.atomic_numbers
        )

        data_for_readout = copy(data)

        data_for_readout["edge_attrs"] = edge_attrs
        data_for_readout["edge_feats"] = edge_feats

        # data._edge_attrs_keys = (*data._edge_attrs_keys, "edge_attrs", "edge_feats")

        # Apply the readouts.
        if not self.readout_per_interaction:
            # Readout from the whole set of features
            node_labels, edge_labels = self.matrix_readouts(
                data=data_for_readout,
                node_feats=mace_out["node_feats"],
            )
        else:
            # Go interaction by interaction and grab the features that each one produced
            # Apply the readout to each interaction and then sum them all.
            used = 0
            node_labels_list = []
            edge_labels_list = []
            for i, readout in enumerate(self.matrix_readouts):
                inter_dim = self.mace_inter_irreps[i].dim
                inter_node_feats = mace_out["node_feats"][:, used : used + inter_dim]
                used += inter_dim

                node_labels, edge_labels = readout(
                    data=data_for_readout,
                    node_feats=inter_node_feats,
                )

                node_labels_list.append(node_labels)
                edge_labels_list.append(edge_labels)

            node_labels = torch.stack(node_labels_list).sum(axis=0)
            edge_labels = torch.stack(edge_labels_list).sum(axis=0)

        return {**mace_out, "node_labels": node_labels, "edge_labels": edge_labels}
