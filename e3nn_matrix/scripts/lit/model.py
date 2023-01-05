from pathlib import Path
from typing import Type, Union

import pytorch_lightning as pl
import torch

#from context import mace
from e3nn_matrix.data.metrics import OrbitalMatrixMetric, block_type_mse
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges

class LitOrbitalMatrixModel(pl.LightningModule):
    model: torch.nn.Module
    
    def __init__(
        self,
        root_dir: str = ".",
        basis_files: Union[str, None] = None,
        z_table: Union[AtomicTableWithEdges, None] = None,
        loss: Type[OrbitalMatrixMetric] = block_type_mse,
        **kwargs
        ):

        super().__init__()

        self.save_hyperparameters()

        if z_table is None:
            self.z_table = AtomicTableWithEdges.from_basis_glob(Path(root_dir).glob(basis_files))
        else:
            self.z_table = z_table

        self.loss_fn = loss()

        self.model = None # Here the model should be initialized.

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        out = self.model(batch)

        loss, stats = self.loss_fn(
            nodes_pred=out['node_labels'], nodes_ref=batch['atom_labels'],
            edges_pred=out['edge_labels'], edges_ref=batch['edge_labels']
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for k, v in stats.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)

        loss, stats = self.loss_fn(
            nodes_pred=out['node_labels'], nodes_ref=batch['atom_labels'],
            edges_pred=out['edge_labels'], edges_ref=batch['edge_labels']
        )

        self.log("val_loss", loss, prog_bar=True, logger=True)

        for k, v in stats.items():
            self.log(f"val_{k}", v)

        return out