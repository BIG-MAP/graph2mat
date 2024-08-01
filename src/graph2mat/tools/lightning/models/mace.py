from typing import Type, Union, Optional

from e3nn import o3
from mace.modules import MACE, InteractionBlock, RealAgnosticResidualInteractionBlock
import torch

# from context import mace
from graph2mat.core.data.metrics import OrbitalMatrixMetric, block_type_mse
from graph2mat import BasisTableWithEdges

from graph2mat.bindings.e3nn import (
    E3nnSimpleNodeBlock,
    E3nnSimpleEdgeBlock,
    E3nnGraph2Mat,
    E3nnInteraction,
    E3nnEdgeMessageBlock,
)

from graph2mat.models.mace import MatrixMACE

# from graph2mat.models._mace.models import OrbitalMatrixMACE

from graph2mat.tools.lightning import LitBasisMatrixModel


class LitMACEMatrixModel(LitBasisMatrixModel):
    model: MACE

    def __init__(
        self,
        root_dir: str = ".",
        basis_files: Union[str, None] = None,
        basis_table: Union[BasisTableWithEdges, None] = None,
        no_basis: Optional[dict] = None,
        # r_max: float=3.0,
        num_bessel: int = 10,
        num_polynomial_cutoff: int = 3,
        max_ell: int = 3,
        interaction_cls: Type[InteractionBlock] = RealAgnosticResidualInteractionBlock,
        interaction_cls_first: Type[
            InteractionBlock
        ] = RealAgnosticResidualInteractionBlock,
        num_interactions: int = 2,
        # num_elements: int=2,
        hidden_irreps: Union[o3.Irreps, str] = "20x0e+20x1o+20x2e",
        edge_hidden_irreps: Union[o3.Irreps, str] = "4x0e+4x1o+4x2e",
        avg_num_neighbors: float = 1.0,
        # atomic_numbers: List[int],
        correlation: int = 1,
        # unique_atoms: Sequence[sisl.Atom],
        symmetric_matrix: bool = False,
        preprocessing_nodes: Optional[Type[torch.nn.Module]] = None,
        preprocessing_edges: Optional[Type[torch.nn.Module]] = None,
        node_block_readout: Type[torch.nn.Module] = E3nnSimpleNodeBlock,
        edge_block_readout: Type[torch.nn.Module] = E3nnSimpleEdgeBlock,
        readout_per_interaction: bool = False,
        optim_wdecay: float = 5e-7,
        optim_amsgrad: bool = True,
        optim_lr: float = 1e-3,
        loss: Type[OrbitalMatrixMetric] = block_type_mse,
        initial_node_feats: str = "OneHotZ",
        version: str = "new",
    ):
        model_cls = MatrixMACE  # if version == "new" else OrbitalMatrixMACE

        super().__init__(
            root_dir=root_dir,
            basis_files=basis_files,
            basis_table=basis_table,
            no_basis=no_basis,
            loss=loss,
            initial_node_feats="OneHotZ",
            model_cls=model_cls,
        )
        self.save_hyperparameters()

        if isinstance(hidden_irreps, str):
            hidden_irreps = o3.Irreps(hidden_irreps)
        if isinstance(edge_hidden_irreps, str):
            edge_hidden_irreps = o3.Irreps(edge_hidden_irreps)

        if version == "new":
            mace = MACE(
                r_max=self.basis_table.maxR() * 2,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                max_ell=max_ell,
                interaction_cls=interaction_cls,
                interaction_cls_first=interaction_cls_first,
                num_interactions=num_interactions,
                num_elements=len(self.basis_table.basis),
                hidden_irreps=hidden_irreps,
                MLP_irreps=o3.Irreps("1x0e"),
                atomic_energies=torch.zeros(len(self.basis_table.basis)),
                avg_num_neighbors=avg_num_neighbors,
                atomic_numbers=torch.arange(len(self.basis_table.basis)),
                correlation=correlation,
                gate=None,
            )

            self.init_model(
                mace=mace,
                readout_per_interaction=readout_per_interaction,
                unique_basis=self.basis_table.basis,
                edge_hidden_irreps=edge_hidden_irreps,
                symmetric=symmetric_matrix,
                preprocessing_nodes=preprocessing_nodes,
                preprocessing_edges=preprocessing_edges,
                node_operation=node_block_readout,
                edge_operation=edge_block_readout,
            )

        else:
            self.init_model(
                r_max=self.basis_table.maxR() * 2,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                max_ell=max_ell,
                interaction_cls=interaction_cls,
                interaction_cls_first=interaction_cls_first,
                num_interactions=num_interactions,
                num_elements=len(self.basis_table.basis),
                hidden_irreps=hidden_irreps,
                edge_hidden_irreps=edge_hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                correlation=correlation,
                unique_basis=self.basis_table.basis,
                matrix_readout=E3nnGraph2Mat,
                symmetric_matrix=symmetric_matrix,
                node_block_readout=node_block_readout,
                edge_block_readout=edge_block_readout,
                only_last_readout=False,
                node_attr_irreps=self.initial_node_feats_irreps,
            )

    def configure_optimizers(self):
        model = self.model

        # Some parameters of the optimizer saved in the module.
        weight_decay = self.hparams.optim_wdecay
        amsgrad = self.hparams.optim_amsgrad

        # Optimizer setup used by the developers of MACE:
        decay_interactions = {}
        no_decay_interactions = {}
        for name, param in model.mace.interactions.named_parameters():
            if "linear.weight" in name or "skip_tp_full.weight" in name:
                decay_interactions[name] = param
            else:
                no_decay_interactions[name] = param

        param_options = dict(
            params=[
                {
                    "name": "embedding",
                    "params": model.mace.node_embedding.parameters(),
                    "weight_decay": 0.0,
                },
                {
                    "name": "interactions_decay",
                    "params": list(decay_interactions.values()),
                    "weight_decay": weight_decay,
                },
                {
                    "name": "interactions_no_decay",
                    "params": list(no_decay_interactions.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "products",
                    "params": model.mace.products.parameters(),
                    "weight_decay": weight_decay,
                },
                {
                    "name": "readouts",
                    "params": model.matrix_readouts.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=self.hparams.optim_lr,
            amsgrad=amsgrad,
        )

        return torch.optim.Adam(**param_options)
