from pathlib import Path
from typing import Type, Union, Optional
import warnings

import pytorch_lightning as pl
import torch

#from context import mace
from e3nn_matrix.data.metrics import OrbitalMatrixMetric, block_type_mse
from e3nn_matrix.torch.data import OrbitalMatrixData
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
            if basis_files is None:
                self.z_table = None
            else:
                self.z_table = AtomicTableWithEdges.from_basis_glob(Path(root_dir).glob(basis_files))
        else:
            self.z_table = z_table

        self.loss_fn = loss()

        self.model = None # Here the model should be initialized.

    def forward(self,
        x:OrbitalMatrixData,
        calculate_forces=False,
        calculate_grads=False,
        hamiltonian: Optional[OrbitalMatrixData]=None,
    ):

        # If we need to calculate forces or grads,
        # activate gradient tracking on xyz positions
        orig_state = x.positions.requires_grad
        if calculate_forces or calculate_grads:
            x.positions.requires_grad_(True)

        pred = self.model(x)
        if calculate_forces:
            node_entries = pred["node_labels"]
            edge_entries = pred["edge_labels"]
            if hamiltonian is not None:
                node_entries = node_entries*hamiltonian.atom_labels
                edge_entries = edge_entries*hamiltonian.edge_labels

            node_force_contrib = torch.autograd.grad(
                node_entries,
                x.positions,
                retain_graph=True, # Do not free graph yet
                grad_outputs=torch.ones_like(node_entries),
            )[0]
            edge_force_contrib = torch.autograd.grad(
                edge_entries,
                x.positions,
                retain_graph=calculate_grads, # We still need the graph for more calculations
                grad_outputs=torch.ones_like(edge_entries),
            )[0]
            total_force = edge_force_contrib + node_force_contrib
            # Invert to original basis
            # TODO: Not sure if this should be done here
            total_force = total_force @ x._inv_change_of_basis.T

            pred["forces"] = total_force

        if calculate_grads:
            node_entries = pred["node_labels"]
            edge_entries = pred["edge_labels"]
            node_grads = torch.autograd.grad(
                node_entries,
                x.positions,
                grad_outputs=torch.diag(torch.ones_like(node_entries)),
                retain_graph=True,
                is_grads_batched=True,
            )[0]
            edge_grads = torch.autograd.grad(
                edge_entries,
                x.positions,
                grad_outputs=torch.diag(torch.ones_like(edge_entries)),
                is_grads_batched=True,
            )[0]
            # Invert to original basis
            # TODO: Not sure if this should be done here
            pred["node_grads"] = node_grads @ x._inv_change_of_basis.T
            pred["edge_grads"] = edge_grads @ x._inv_change_of_basis.T

        x.positions.requires_grad_(orig_state)

        return pred

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

    def on_load_checkpoint(self, checkpoint) -> None:
        "Objects to retrieve from checkpoint file"
        try:
            self.z_table = checkpoint["z_table"]
        except KeyError:
            warnings.warn("Failed to load z_table from checkpoint: Key does not exist.")
