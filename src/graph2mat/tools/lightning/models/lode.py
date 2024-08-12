from typing import Type, Union, Optional, Literal

from e3nn import o3
import torch

# from context import mace
from graph2mat.core.data.metrics import OrbitalMatrixMetric, block_type_mse
from graph2mat import BasisTableWithEdges

from graph2mat.bindings.e3nn import (
    E3nnSimpleNodeBlock,
    E3nnSimpleEdgeBlock,
)

from graph2mat.models.lode import MatrixE3nnLODE
from graph2mat.bindings.salted import E3nnLODE

# from graph2mat.models._mace.models import OrbitalMatrixMACE

from graph2mat.tools.lightning import LitBasisMatrixModel


class LitLODEMatrixModel(LitBasisMatrixModel):
    model: E3nnLODE

    def __init__(
        self,
        root_dir: str = ".",
        basis_files: Union[str, None] = None,
        basis_table: Union[BasisTableWithEdges, None] = None,
        no_basis: Optional[dict] = None,
        # LODE parameters
        lode_lmax: int = 1,
        lode_rep1: Literal["rho", "V"] = "rho",
        lode_rep1_params: dict = {
            "cutoff": 1.5,
            "max_radial": 5,
            "max_angular": 2,
            "atomic_gaussian_width": 1.2,
        },
        lode_rep2: Literal["rho", "V"] = "V",
        lode_rep2_params: dict = {
            "cutoff": 1.5,
            "max_radial": 5,
            "max_angular": 2,
            "atomic_gaussian_width": 1.2,
        },
        # num_elements: int=2,
        node_hidden_irreps: Union[o3.Irreps, str] = "20x0e+20x1o+20x2e",
        # unique_atoms: Sequence[sisl.Atom],
        symmetric_matrix: bool = False,
        preprocessing_nodes: Optional[Type[torch.nn.Module]] = None,
        preprocessing_edges: Optional[Type[torch.nn.Module]] = None,
        node_block_readout: Type[torch.nn.Module] = E3nnSimpleNodeBlock,
        edge_block_readout: Type[torch.nn.Module] = E3nnSimpleEdgeBlock,
        loss: Type[OrbitalMatrixMetric] = block_type_mse,
        initial_node_feats: str = "OneHotZ",
        optim_wdecay: float = 5e-7,
        optim_amsgrad: bool = True,
        optim_lr: float = 1e-3,
    ):
        model_cls = MatrixE3nnLODE  # if version == "new" else OrbitalMatrixMACE

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

        if isinstance(node_hidden_irreps, str):
            node_hidden_irreps = o3.Irreps(node_hidden_irreps)

        lode = E3nnLODE(
            self.basis_table.basis,
            lmax=lode_lmax,
            rep1=lode_rep1,
            params_rep1=lode_rep1_params,
            rep2=lode_rep2,
            # rep2="rho",
            params_rep2=lode_rep2_params,
        )

        self.init_model(
            lode=lode,
            unique_basis=self.basis_table.basis,
            node_hidden_irreps=node_hidden_irreps,
            # edge_hidden_irreps=edge_hidden_irreps,
            symmetric=symmetric_matrix,
            preprocessing_nodes=preprocessing_nodes,
            preprocessing_edges=preprocessing_edges,
            node_operation=node_block_readout,
            edge_operation=edge_block_readout,
        )

    def configure_optimizers(self):
        model = self.model

        return torch.optim.Adam(
            params=model.parameters(),
            lr=self.hparams.optim_lr,
            amsgrad=self.hparams.optim_amsgrad,
        )
