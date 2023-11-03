from typing import Type, Union

from e3nn import o3
import torch

#from context import mace
from e3nn_matrix.data.metrics import OrbitalMatrixMetric, block_type_mse
from e3nn_matrix.data.table import BasisTableWithEdges
from e3nn_matrix.torch.modules import NodeBlock, EdgeBlock, SimpleNodeBlock, SimpleEdgeBlock
from e3nn_matrix.torch.modules.mace import MACEBasisMatrixReadout


from e3nn_matrix.models.mace import OrbitalMatrixMACE

from e3nn_matrix.tools.lightning import LitBasisMatrixModel

from mace.modules.blocks import InteractionBlock, RealAgnosticResidualInteractionBlock

class LitOrbitalMatrixMACE(LitBasisMatrixModel):
    model: OrbitalMatrixMACE
    
    def __init__(
        self,
        root_dir: str = ".",
        basis_files: Union[str, None] = None,
        basis_table: Union[BasisTableWithEdges, None] = None,
        #r_max: float=3.0,
        num_bessel: int=10,
        num_polynomial_cutoff: int=3,
        max_ell: int=3,
        interaction_cls: Type[InteractionBlock]=RealAgnosticResidualInteractionBlock,
        interaction_cls_first: Type[InteractionBlock]=RealAgnosticResidualInteractionBlock,
        num_interactions: int=2,
        #num_elements: int=2,
        hidden_irreps: Union[o3.Irreps, str]="20x0e+20x1o+20x2e",
        edge_hidden_irreps: Union[o3.Irreps, str]="4x0e+4x1o+4x2e",
        avg_num_neighbors: float=1.,
        #atomic_numbers: List[int],
        correlation: int=1,
        #unique_atoms: Sequence[sisl.Atom],
        matrix_readout: Type[MACEBasisMatrixReadout] = MACEBasisMatrixReadout,
        symmetric_matrix: bool = False,
        node_block_readout: Type[NodeBlock] = SimpleNodeBlock,
        edge_block_readout: Type[EdgeBlock] = SimpleEdgeBlock,
        only_last_readout: bool = False,
        optim_wdecay: float=5e-7,
        optim_amsgrad: bool=True,
        optim_lr: float=1e-3,
        loss: Type[OrbitalMatrixMetric] = block_type_mse,
        ):

        super().__init__(root_dir=root_dir, basis_files=basis_files, basis_table=basis_table, loss=loss, model_cls=OrbitalMatrixMACE)
        self.save_hyperparameters()

        if isinstance(hidden_irreps, str):
            hidden_irreps = o3.Irreps(hidden_irreps)
        if isinstance(edge_hidden_irreps, str):
            edge_hidden_irreps = o3.Irreps(edge_hidden_irreps)

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
            matrix_readout=matrix_readout,
            symmetric_matrix=symmetric_matrix,
            node_block_readout=node_block_readout,
            edge_block_readout=edge_block_readout,
            only_last_readout=only_last_readout
        )

    def configure_optimizers(self):

        model = self.model

        # Some parameters of the optimizer saved in the module.
        weight_decay = self.hparams.optim_wdecay
        amsgrad = self.hparams.optim_amsgrad

        # Optimizer setup used by the developers of MACE:
        decay_interactions = {}
        no_decay_interactions = {}
        for name, param in model.interactions.named_parameters():
            if "linear.weight" in name or "skip_tp_full.weight" in name:
                decay_interactions[name] = param
            else:
                no_decay_interactions[name] = param

        param_options = dict(
            params=[
                {
                    "name": "embedding",
                    "params": model.node_embedding.parameters(),
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
                    "params": model.products.parameters(),
                    "weight_decay": weight_decay,
                },
                {
                    "name": "readouts",
                    "params": model.readouts.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=self.hparams.optim_lr,
            amsgrad=amsgrad,
        )

        return torch.optim.Adam(**param_options)
