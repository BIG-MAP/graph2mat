"""Pytorch_lightning callbacks for I/O, progress tracking and visualization, etc..."""
from pathlib import Path
from typing import Sequence, Union, Literal, Dict, Any, Type
import io
import csv

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
import sisl
import numpy as np

from e3nn_matrix.data.batch_utils import batch_to_orbital_matrix_data
from e3nn_matrix.data.sparse import nodes_and_edges_to_sparse_orbital
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges
from e3nn_matrix.data.metrics import OrbitalMatrixMetric
from e3nn_matrix.tools.viz import plot_orbital_matrix

class MatrixWriter(BasePredictionWriter):
    """Callback to write predicted matrices to disk."""
    def __init__(self, output_file: str, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.output_file = output_file

    def write_on_batch_end(self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Dict,
        batch_indices: Sequence[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int):

        # Get the atomic table, which will help us constructing the matrices from
        # the batched flat arrays.
        z_table: AtomicTableWithEdges = trainer.datamodule.z_table
        # Find out whether the model was trained with the imposition that the matrix
        # is symmetric.
        symmetric_matrix = trainer.datamodule.symmetric_matrix

        # Find out which matrix class we should use based on what matrix type the data
        # has been trained on.
        matrix_cls = {
            "density_matrix": sisl.DensityMatrix,
            "energy_density_matrix": sisl.EnergyDensityMatrix,
            "hamiltonian": sisl.Hamiltonian,
            "dynamical_matrix": sisl.DynamicalMatrix,
        }[trainer.datamodule.out_matrix]

        # Get iterator to loop through matrices in batch
        matrix_iter = batch_to_orbital_matrix_data(
            batch,
            prediction,
            z_table,
            symmetric_matrix,
        )

        # Loop through structures in the batch
        for matrix_data in matrix_iter:
            # Get the path from which this structure was read.
            path = matrix_data.metadata["path"]
            sparse_orbital_matrix = matrix_data.to_sparse_orbital_matrix(z_table, matrix_cls, symmetric_matrix, trainer.datamodule.sub_atomic_matrix)

            # And write the matrix to it.
            sparse_orbital_matrix.write(path.parent / self.output_file)


class SamplewiseMetricsLogger(Callback):
    """Creates a CSV file with multiple metrics for each sample of the dataset.
    
    This callback is needed because otherwise the metrics are computed and logged
    on a per-batch basis.

    Each row of the CSV file is a computation of all metrics for a single sample on a given epoch.
    Therefore, the csv file contains the following columns: [sample_name, ...metrics..., split_key, epoch_index]

    Parameters
    ----------
    metrics : Sequence[Type[OrbitalMatrixMetric]]
        List of metrics to compute.
    splits : Sequence[str], optional
        List of splits for which to compute the metrics. Can be any combination of "train", "val", "test".
    output_file : Union[str, Path], optional
        Path to the output CSV file.
    """

    def __init__(self,
        metrics: Union[Sequence[Type[OrbitalMatrixMetric]], None] = None,
        splits: Sequence = ["train", "val", "test"], # I don't know why, but Sequence[str] breaks the lightning CLI
        output_file: Union[str, Path] = "sample_metrics.csv"
    ):
        super().__init__()

        if splits in ["train", "val", "test"]:
            splits = [splits]
        elif isinstance(splits, str):
            raise ValueError(f"Invalid value for splits: {splits}")
        
        if metrics is None:
            metrics = OrbitalMatrixMetric.__subclasses__()

        self.splits = splits
        self.metrics = [metric() for metric in metrics]
        self.output_file = output_file

        self._init_file()

    def on_train_epoch_start(self, trainer, pl_module):
        if "train" in self.splits:
            self._on_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        if "val" in self.splits:
            self._on_epoch_start(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        if "test" in self.splits:
            self._on_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if "train" in self.splits:
            self._on_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val" in self.splits:
            self._on_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if "test" in self.splits:
            self._on_epoch_end(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if "train" in self.splits:
            self._on_batch_end("train", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if "val" in self.splits:
            self._on_batch_end("val", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if "test" in self.splits:
            self._on_batch_end("test", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def _on_epoch_start(self, trainer, pl_module):
        self.open_file_handle()

    def _on_batch_end(self, split, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # Get the atomic table, which some metrics might need.
        z_table: AtomicTableWithEdges = trainer.datamodule.z_table

        # Compute all the metrics
        metrics = [
            metric(
                nodes_pred=outputs['node_labels'], nodes_ref=batch.atom_labels, 
                edges_pred=outputs['edge_labels'], edges_ref=batch.edge_labels, batch=batch,
                z_table=z_table, config_resolved=True, 
                symmetric_matrix=trainer.datamodule.symmetric_matrix,
            )[0] for metric in self.metrics
        ]

        # Concatenate them (if we wish to accumulate them) TODO

        # Get the index of the current epoch
        current_epoch = trainer.current_epoch
        # And the names of the samples that we are going to log
        sample_names = [Path(metadata["path"]).parent.name for metadata in batch.metadata]

        # Create an iterator that will return the data to be written to the CSV file for each row.
        # That is, first the sample name, then the metrics, and finally the split and the epoch index.
        iterator = ([sample_names[i], *data, split, current_epoch] for i, data in enumerate(np.array(metrics).T))

        # Write the data
        self.csv_writer.writerows(iterator)
    
    def _on_epoch_end(self, trainer, pl_module):
        self.close_file_handle()

    def _init_file(self):
        # Open the file
        with open(self.output_file, "w", newline="") as csv_file:
            # And write the headers, i.e. column names
            metric_names=[metric.__class__.__name__ for metric in self.metrics]
            fieldnames = ["sample_name", *metric_names, "split", "epoch"]

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(fieldnames)

    def open_file_handle(self):
        # Open the file
        self.output_fd = open(self.output_file, "a", newline="")
        self.csv_writer = csv.writer(self.output_fd)

    def close_file_handle(self):
        self.output_fd.close()    


class PlotMatrixValidationError(Callback):
    """Add plots of MAE and RMSE for each entry of matrix.
    Does only work if the matrix is the same format for every datapoint as in molecular dynamics data"""
    def __init__(self):
        super().__init__()
    def on_validation_epoch_start(self, trainer, pl_module):
        self.node_running_ae = None
        self.node_running_se = None
        self.edge_running_ae = None
        self.edge_running_se = None
        self.matrix_count = 0
        self.positions = None
        self.atom_types = None
        self.cell = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # And the atomic table, which will help us constructing the matrices from
        # the batched flat arrays.
        z_table: AtomicTableWithEdges = trainer.datamodule.z_table
        # Find out whether the model was trained with the imposition that the matrix
        # is symmetric.
        symmetric_matrix = trainer.datamodule.symmetric_matrix

        # Pointer arrays to understand where the data for each structure starts in the batch. 
        atom_ptr = batch.ptr.numpy(force=True)
        edge_ptr = np.zeros_like(atom_ptr)
        np.cumsum(batch.n_edges.numpy(force=True), out=edge_ptr[1:])

        # Types for both atoms and edges.
        atom_types = batch.atom_types.numpy(force=True)
        edge_types = batch.edge_types.numpy(force=True)

        # Get the values for the node blocks and the pointer to the start of each block.
        node_labels = outputs['node_labels'].numpy(force=True)
        node_labels_ptr = z_table.atom_block_pointer(atom_types)

        # Get the values for the edge blocks and the pointer to the start of each block.
        edge_index = batch.edge_index.numpy(force=True)
        if symmetric_matrix:
            edge_index = edge_index[:, ::2]
            edge_types = edge_types[::2]
            edge_ptr = edge_ptr // 2

        edge_labels = outputs['edge_labels'].numpy(force=True)
        edge_labels_ptr = z_table.edge_block_pointer(edge_types)

        # Save the relevant information for use when plotting
        # We only save the information for one configuration and assume the
        # rest are the same
        self.positions = batch.positions.numpy(force=True)[atom_ptr[0]:atom_ptr[1]]
        self.cell = batch.cell.numpy(force=True)[0:3]
        self.nsc = batch.nsc[0].numpy(force=True)
        self.atom_types = atom_types[atom_ptr[0]:atom_ptr[1]]
        self.node_labels_ptr = node_labels_ptr[atom_ptr[0]:(atom_ptr[1]+1)]
        self.edge_labels_ptr = edge_labels_ptr[edge_ptr[0]:(edge_ptr[1]+1)]
        self.edge_index = edge_index[:, edge_ptr[0]:edge_ptr[1]]

        # Calculate errors on the flat arrays
        node_error = node_labels - batch['atom_labels'].numpy(force=True)
        edge_error = edge_labels - batch['edge_labels'].numpy(force=True)

        # Some edge labels are np.nan (which means no entry), manually set the error
        # for those to 0.
        edge_error[np.isnan(edge_error)] = 0

        # Sum the errors across the different configurations
        # First get the pointers that split the labels...
        atom_split = atom_ptr[1:-1]
        edge_split = edge_ptr[1:-1]

        edge_labels_split = edge_labels_ptr[edge_split]
        node_labels_split = node_labels_ptr[atom_split]
        np.testing.assert_array_equal(edge_labels_split.shape, node_labels_split.shape)
        # ..and then sum across the configurations
        self.matrix_count += edge_labels_split.shape[0] + 1
        nae, eae, nse, ese = (sum(np.split(np.abs(node_error), node_labels_split)),
                             sum(np.split(np.abs(edge_error), edge_labels_split)),
                             sum(np.split(np.square(node_error), node_labels_split)),
                             sum(np.split(np.square(edge_error), edge_labels_split)),)
        self.node_running_ae = nae + self.node_running_ae if self.node_running_ae is not None else nae
        self.edge_running_ae = eae + self.edge_running_ae if self.edge_running_ae is not None else eae
        self.node_running_se = nse + self.node_running_se if self.node_running_se is not None else nse
        self.edge_running_se = ese + self.edge_running_se if self.edge_running_se is not None else ese

    def on_validation_epoch_end(self, trainer, pl_module):
        import PIL.Image
        # Find out which matrix class we should use based on what matrix type the data
        # has been trained on.
        matrix_cls = {
            "density_matrix": sisl.DensityMatrix,
            "energy_density_matrix": sisl.EnergyDensityMatrix,
            "hamiltonian": sisl.Hamiltonian,
            "dynamical_matrix": sisl.DynamicalMatrix,
        }[trainer.datamodule.out_matrix]

        # The z_table helps us arrange the errors into matrix format
        z_table: AtomicTableWithEdges = trainer.datamodule.z_table

        labels = ["MAE", "RMSE"]
        node_errors = [self.node_running_ae, self.node_running_se]
        edge_errors = [self.edge_running_ae, self.edge_running_se]

        assert self.atom_types is not None
        for i, (label, node_error, edge_error) in enumerate(zip(labels, node_errors, edge_errors)):
            geometry = sisl.Geometry(
                self.positions,
                atoms=[z_table.atoms[at_type] for at_type in self.atom_types],
                sc=self.cell)

            geometry.set_nsc(self.nsc)

            assert node_error is not None
            assert edge_error is not None
            if label == "RMSE":
                ne = np.sqrt(node_error/self.matrix_count)
                ee = np.sqrt(edge_error/self.matrix_count)
            else:
                ne = node_error/self.matrix_count
                ee = edge_error/self.matrix_count

            # Convert flat error array to matrix format
            matrix = nodes_and_edges_to_sparse_orbital(
                node_vals=ne, node_ptr=self.node_labels_ptr,
                edge_vals=ee, edge_index=self.edge_index,
                edge_ptr=self.edge_labels_ptr,
                geometry=geometry, sp_class=matrix_cls, symmetrize_edges=trainer.datamodule.symmetric_matrix).tocsr()

            # Plot image to figure object
            fig = plot_orbital_matrix(
                matrix, geometry=geometry, atom_lines=True, basis_lines=True, text=".3f"
            )

            # Convert image to png
            img_bytes = fig.to_image(format="png", width=800, height=600)
            img_buf = io.BytesIO(img_bytes)
            img = PIL.Image.open(img_buf, formats=['PNG'])
            # Convert to numpy array
            img_array = np.array(img)
            # Add to tensorboard
            tensorboard = trainer.logger.experiment
            tensorboard.add_image(label, img_array, dataformats='HWC', global_step=trainer.global_step)
