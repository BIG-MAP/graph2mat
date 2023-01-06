import json
import math
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
        test_runs: Optional[str] = None,
        predict_structs: Optional[str] = None,
        runs_json: Optional[str] = None,
        symmetric_matrix: bool = False,
        sub_atomic_matrix: bool = True,
        batch_size: int = 5,
        loader_threads: int=1,
    ):
        """

        Parameters
        ----------
            out_matrix : Literal['density_matrix', 'hamiltonian', 'energy_density_matrix', 'dynamical_matrix']
            basis_files : Union[str, None]
            z_table : Union[AtomicTableWithEdges, None]
            root_dir : str
            train_runs : Optional[str]
            val_runs : Optional[str]
            test_runs : Optional[str]
            predict_structs : Optional[str]
            runs_json: Optional[str]
                Path to json-file with a dictionary where the keys are train/val/test/predict.
                and the dictionary values are list of paths to the run files relative to `root_dir`
                The paths will be overwritten by train_runs/val_runs/test_runs/predict_structs if given.
            symmetric_matrix : bool
            sub_atomic_matrix : bool
            batch_size : int
            loader_threads : int

        """
        super().__init__()
        self.save_hyperparameters()

        self.root_dir = root_dir

        self.basis_files = basis_files
        self.z_table = z_table
        self.out_matrix = out_matrix
        self.symmetric_matrix = symmetric_matrix

        self.train_runs = train_runs
        self.val_runs = val_runs
        self.test_runs = test_runs
        self.runs_json = runs_json

        self.predict_runs = predict_structs
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


        # Load json file with paths for each split
        if self.runs_json is not None:
            with open(self.runs_json, "r") as f:
                runs_dict = json.load(f)
        else:
            runs_dict = {}

        # Set the paths for each split
        for split in ["train", "val", "test", "predict"]:
            glob_path = getattr(self, "%s_runs" % split)
            # Use the glob paths if given
            if glob_path is not None:
                runs = Path(self.root_dir).glob(glob_path)
            # Else use the json file
            elif split in runs_dict:
                runs = [Path(self.root_dir) / p for p in runs_dict[split]]
            else:
                runs = None

            if runs is not None:
                # For predictions we do not have a matrix to read
                if split == "predict":
                    out_matrix = None
                else:
                    out_matrix = self.out_matrix
                # Read the data
                data = [
                    OrbitalMatrixData.from_config(
                        load_orbital_config_from_run(run, out_matrix=out_matrix),
                        z_table=self.z_table, sub_atomic_matrix=self.sub_atomic_matrix,
                        symmetric_matrix=self.symmetric_matrix
                    )
                    for run in runs
                ]
                setattr(self, "%s_data" % split, data)

        # Now check if we should take some data out of the training data to use it for
        # validation.
        if self.val_data is None and self.train_data is not None:
            # Instantiate a random state to sample from data
            rng = np.random.RandomState(32)
            # User didn't specify a validation directory, just randomly draw 10 percent from the training.
            perms = rng.permutation(len(self.train_data))
            num_val = int(math.ceil(len(self.train_data)/10))

            self.val_data = [self.train_data[x] for x in np.sort(perms[0:num_val])]
            self.train_data = [self.train_data[x] for x in np.sort(perms[num_val:])]

    def train_dataloader(self):
        assert self.train_data is not None, "No training data was provided, please set the ``train_runs`` argument."
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.hparams.loader_threads)

    def val_dataloader(self):
        assert self.val_data is not None, "No validation data was provided, please set either the ``train_runs`` or the ``val_runs`` argument."
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.hparams.loader_threads)

    def test_dataloader(self):
        assert self.test_data is not None, "No test data was provided, please set the ``test_runs`` argument."
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.hparams.loader_threads)

    def predict_dataloader(self):
        assert self.predict_data is not None, "No prediction data was provided, please set the ``predict_structs`` argument."
        return DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.hparams.loader_threads)
