from pathlib import Path
from typing import Type, Union
import warnings

import pytorch_lightning as pl
import torch

#from context import mace
from e3nn_matrix.data.metrics import OrbitalMatrixMetric, block_type_mse
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges
from e3nn_matrix import __version__

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
            if basis_files is None:
                self.z_table = None
            else:
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
            edges_pred=out['edge_labels'], edges_ref=batch['edge_labels'],
            log_verbose=True
        )

        self.log("val_loss", loss, prog_bar=True, logger=True)
        # save validation loss as the hyperparameter opt metric (used by tensorboard)
        self.log("hp_metric", loss)

        for k, v in stats.items():
            self.log(f"val_{k}", v)

        return out

    def test_step(self, batch, batch_idx):
        out = self.model(batch)

        loss, stats = self.loss_fn(
            nodes_pred=out['node_labels'], nodes_ref=batch['atom_labels'],
            edges_pred=out['edge_labels'], edges_ref=batch['edge_labels']
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)

        for k, v in stats.items():
            self.log(f"test_{k}", v)

        return out

    def on_save_checkpoint(self, checkpoint) -> None:
        "Objects to include in checkpoint file"
        checkpoint["z_table"] = self.z_table
        checkpoint["version"] = __version__

    def on_load_checkpoint(self, checkpoint) -> None:
        "Objects to retrieve from checkpoint file"
        try:
            self.z_table = checkpoint["z_table"]
        except KeyError:
            warnings.warn("Failed to load z_table from checkpoint: Key does not exist.")

        try:
            ckpt_version = checkpoint["version"]
        except KeyError:
            ckpt_version = None
            warnings.warn("Unable to determine version that created checkpoint file")
        if ckpt_version:
            if not (ckpt_version == __version__):
                warnings.warn("The checkpoint version %s does not match the current package version %s" % (ckpt_version, __version__))

