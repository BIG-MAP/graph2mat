"""Pytorch_lightning callbacks for I/O, progress tracking and visualization, etc..."""
from pathlib import Path
from typing import Sequence, Union, Literal, Dict, Any
import io

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
import sisl
import numpy as np

from e3nn_matrix.data.batch_utils import batch_to_sparse_orbital, batch_to_orbital_matrix_data
from e3nn_matrix.data.sparse import nodes_and_edges_to_sparse_orbital
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges
from e3nn_matrix.viz import plot_orbital_matrix

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

class ComputeNormalizedError(Callback):
    def __init__(self, split: Literal["train", "val", "test"]="test", output_file:Union[Path,str,None]=None, grid_spacing: float=0.1, output_single_file: bool=False):
        """
        Parameters
        ----------
            split : Literal['train', 'val', 'test'] Select which split to calculate error on
            output_file : Filename to write result to
            grid_spacing : float, Grid spacing in Angstrom used for numerical integration
        Raises
        ------
            NotImplementedError : If output_file is given
        """
        super().__init__()
        self.output_file = output_file
        self.split = split
        self.grid_spacing = grid_spacing
        self.output_single_file = output_single_file
        self._reset_counters()

    def _reset_counters(self):
        self.total_error = 0.0
        self.total_electrons = 0.0
        self.per_config_total_error = 0.0
        self.total_num_configs = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        if self.split == "train":
            self._on_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.split == "val":
            self._on_epoch_start(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        if self.split == "test":
            self._on_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.split == "train":
            self._on_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.split == "val":
            self._on_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.split == "test":
            self._on_epoch_end(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.split == "train":
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.split == "val":
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.split == "test":
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # Get the list of paths from which the batch was generated.
        #paths = trainer.datamodule.predict_paths
        # And the atomic table, which will help us constructing the matrices from
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

        matrix_iter_pred = batch_to_orbital_matrix_data(
            batch,
            outputs,
            z_table,
            symmetric_matrix,
        )
        matrix_iter_ref = batch_to_orbital_matrix_data(
            batch,
        )

        # Get indices of data examples of current batch
        #batch_indices = trainer.predict_loop.epoch.current_batch_indices
        # Loop through structures in the batch
        for matrix_pred, matrix_ref in zip( matrix_iter_pred, matrix_iter_ref):
            sp_matrix_pred = matrix_pred.to_sparse_orbital_matrix(z_table, matrix_cls, symmetric_matrix, trainer.datamodule.sub_atomic_matrix)
            sp_matrix_ref = matrix_ref.to_sparse_orbital_matrix(z_table, matrix_cls, symmetric_matrix, trainer.datamodule.sub_atomic_matrix)

            grid_pred = sisl.Grid(self.grid_spacing, geometry=sp_matrix_pred.geometry)
            grid_ref = sisl.Grid(self.grid_spacing, geometry=sp_matrix_ref.geometry)
            np.testing.assert_array_equal(grid_pred.shape, grid_ref.shape)

            sp_matrix_pred.density(grid_pred)
            sp_matrix_ref.density(grid_ref)
            grid_abs_error = abs(grid_pred - grid_ref)
            this_config_error = grid_abs_error.grid.sum()
            this_config_electrons = grid_ref.grid.sum()
            this_config_norm_error = this_config_error/this_config_electrons

            # Summarize in global counters
            self.per_config_total_error += this_config_norm_error
            self.total_error += this_config_error
            self.total_electrons += this_config_electrons
            self.total_num_configs += 1


            if self.output_file is not None:
                path = matrix_ref.metadata["path"].parent
                if self.output_single_file:
                    self.output_fd.write("%s,%.9f\n" % (path, this_config_norm_error))
                else:
                    with open(path / self.output_file, "w") as f:
                        f.write("%.9f\n" % this_config_norm_error)

    def _on_epoch_end(self, trainer, pl_module):
        avg_per_config_error = self.per_config_total_error / self.total_num_configs
        avg_error = self.total_error / self.total_electrons

        pl_module.log("%s_avg_per_config_error_percent" % self.split, avg_per_config_error*100, logger=True, on_epoch=True)
        pl_module.log("%s_avg_error_percent" % self.split, avg_error*100, logger=True, on_epoch=True)

        if self.output_file is not None and self.output_single_file:
            self.output_fd.close()

    def _on_epoch_start(self, trainer, pl_module):
        self._reset_counters()
        if self.output_file is not None and self.output_single_file:
            self.output_fd = open(self.output_file, "w")





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
