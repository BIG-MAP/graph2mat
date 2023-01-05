from pathlib import Path
from typing import Union, Optional, Literal

import pytorch_lightning as pl
import numpy as np

from mace.tools.torch_geometric import DataLoader

from e3nn_matrix.data.configuration import load_orbital_config_from_run
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges
from e3nn_matrix.torch.data import OrbitalMatrixData

class MatrixDataModule(pl.LightningDataModule):
    def __init__(self,
        out_matrix: Literal["density_matrix", "hamiltonian", "energy_density_matrix", "dynamical_matrix"],
        basis_files: Union[str, None] = None,
        z_table: Union[AtomicTableWithEdges, None] = None,
        root_dir: str = ".",
        train_runs: Optional[str] = None, 
        val_runs: Optional[str] = None,
        predict_structs: Optional[str] = None,
        symmetric_matrix: bool = False,
        sub_atomic_matrix: bool = True,
        batch_size: int = 5,  
        loader_threads: int=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_dir = root_dir

        self.basis_files = basis_files
        self.z_table = z_table
        self.out_matrix = out_matrix
        self.symmetric_matrix = symmetric_matrix

        self.train_runs = train_runs
        self.val_runs = val_runs

        self.predict_structs = predict_structs
        self.sub_atomic_matrix = sub_atomic_matrix

        self.batch_size = batch_size

    def setup(self, stage:str):
        if self.z_table is None:
            # Read the basis from the basis files provided.
            self.z_table = AtomicTableWithEdges.from_basis_glob(Path(self.root_dir).glob(self.basis_files))

        # Initialize the data.
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None

        # Instantiate a random state to sample from data in case we need it.
        rng = np.random.RandomState(32)

        # Set the training data
        if self.train_runs is not None:

            # Get all the training data
            runs = Path(self.root_dir).glob(self.train_runs)
            train_data = [
                OrbitalMatrixData.from_config(
                    load_orbital_config_from_run(run, out_matrix=self.out_matrix),
                    z_table=self.z_table, sub_atomic_matrix=self.sub_atomic_matrix,
                    symmetric_matrix=self.symmetric_matrix
                )
                for run in runs
            ]

            # Now check if we should also take some data out of the training data to use it for
            # validation.
            if self.val_runs is None:
                # User didn't specify a validation directory, just randomly draw from the training.
                perms = rng.permutation(len(train_data))

                self.train_data = [train_data[x] for x in np.sort(perms[10:-10])]
                self.val_data = [train_data[x] for x in np.sort(perms[0:10])]
                self.test_data = [train_data[x] for x in np.sort(perms[-10:])]
            else:
                # User specified some validation runs, so use all of this runs for training
                self.train_data = train_data

        # Set the validation and testing data
        if self.val_runs is not None:
            # Get list of all validation runs
            val_runs = list(Path(self.root_dir).glob(self.val_runs))

            # Randomly select 20 of them
            perms = rng.permutation(len(val_runs))
            val_runs = [val_runs[x] for x in np.sort(perms[:20])]

            # Get the data for the selected runs.
            val_data = [
                OrbitalMatrixData.from_config(
                    load_orbital_config_from_run(run, out_matrix=self.out_matrix),
                    z_table=self.z_table, sub_atomic_matrix=self.sub_atomic_matrix,
                    symmetric_matrix=self.symmetric_matrix
                )
                for run in val_runs
            ]

            # Split between validation and test.
            self.val_data = val_data[:10]
            self.test_data = val_data[-10:]

        # Set the prediction data
        if self.predict_structs is not None:
            self.predict_paths = list(Path(self.root_dir).glob(self.predict_structs))

            self.predict_data = [
                OrbitalMatrixData.from_config(
                    load_orbital_config_from_run(run, out_matrix=None),
                    z_table=self.z_table, sub_atomic_matrix=self.sub_atomic_matrix,
                    symmetric_matrix=self.symmetric_matrix
                )
                for run in self.predict_paths
            ]   

    def train_dataloader(self):
        assert self.train_data is not None, "No training data was provided, please set the ``train_runs`` argument."
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.hparams.loader_threads)

    def val_dataloader(self):
        assert self.val_data is not None, "No validation data was provided, please set either the ``train_runs`` or the ``val_runs`` argument."
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.hparams.loader_threads)

    def test_dataloader(self):
        assert self.test_data is not None, "No test data was provided, please set either the ``train_runs`` or the ``val_runs`` argument."
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.hparams.loader_threads)

    def predict_dataloader(self):
        assert self.predict_data is not None, "No prediction data was provided, please set the ``predict_structs`` argument."
        return DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.hparams.loader_threads)