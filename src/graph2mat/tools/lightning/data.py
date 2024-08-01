"""Data loading for pytorch_lightning workflows."""
from typing import Type, Union, List

import json
import math
from pathlib import Path
from typing import Optional
import tempfile
import logging
import os
import shutil

import pytorch_lightning as pl
import numpy as np
import torch.utils.data

from torch_geometric.loader.dataloader import DataLoader

from graph2mat.core.data.configuration import PhysicsMatrixType
from graph2mat import BasisTableWithEdges, AtomicTableWithEdges, MatrixDataProcessor
from graph2mat.core.data.node_feats import NodeFeature
from graph2mat.bindings.torch.data import TorchBasisMatrixData
from graph2mat.bindings.torch import (
    TorchBasisMatrixDataset,
    InMemoryData,
    RotatingPoolData,
)


class MatrixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        out_matrix: Optional[PhysicsMatrixType] = None,
        basis_files: Optional[str] = None,
        no_basis: Optional[dict] = None,
        basis_table: Optional[BasisTableWithEdges] = None,
        root_dir: str = ".",
        train_runs: Optional[str] = None,
        val_runs: Optional[str] = None,
        test_runs: Optional[str] = None,
        predict_structs: Optional[str] = None,
        runs_json: Optional[str] = None,
        symmetric_matrix: bool = False,
        sub_point_matrix: bool = True,
        batch_size: int = 5,
        loader_threads: int = 1,
        copy_root_to_tmp: bool = False,
        store_in_memory: bool = False,
        rotating_pool_size: Optional[int] = None,
        initial_node_feats: str = "OneHotZ",
    ):
        """

        Parameters
        ----------
            out_matrix :'density_matrix', 'hamiltonian', 'energy_density_matrix', 'dynamical_matrix'
            basis_files : Union[str, None]
            basis_table : Union[BasisTableWithEdges, None]
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
            sub_point_matrix : bool
            batch_size : int
            loader_threads : int
            copy_root_to_tmp: bool
            store_in_memory: bool
                If true, will load the dataset into host memory, otherwise it will be read from disk
            rotating_pool_size: int
                If given, the training data will be continously loaded into a smaller poool of this
                size. The data in the active pool can be used several times before it is swapped out
                with new data. This is useful if the data loading is slow. Note that the notion of
                epochs will not be meaningful when this kind of loading is used.
                This will not affect test/val/predict data.

        """
        super().__init__()
        self.save_hyperparameters()

        self.root_dir = root_dir

        self.basis_files = basis_files
        self.no_basis = no_basis
        self.basis_table = basis_table
        self.out_matrix: Optional[PhysicsMatrixType] = out_matrix
        self.symmetric_matrix = symmetric_matrix
        self.initial_node_feats = [
            NodeFeature.registry[k] for k in initial_node_feats.split(" ")
        ]

        self.train_runs = train_runs
        self.val_runs = val_runs
        self.test_runs = test_runs
        self.runs_json = runs_json

        self.predict_runs = predict_structs
        self.sub_point_matrix = sub_point_matrix

        self.batch_size = batch_size
        self.copy_root_to_tmp = copy_root_to_tmp
        self.store_in_memory = store_in_memory
        self.rotating_pool_size = rotating_pool_size
        self.prepare_data_per_node = True
        if self.copy_root_to_tmp:
            self.tmp_dir = Path(tempfile.gettempdir()) / "e3nn_matrix"
        else:
            self.tmp_dir = None

    def prepare_data(self):
        if self.copy_root_to_tmp:
            assert self.tmp_dir is not None
            os.makedirs(self.tmp_dir)
            logging.info("copying %s to %s" % (self.root_dir, self.tmp_dir))
            shutil.copytree(self.root_dir, self.tmp_dir, dirs_exist_ok=True)

    def teardown(self, stage: str):
        if self.tmp_dir is not None:
            logging.info("deleting dir %s" % (self.tmp_dir))
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def setup(self, stage: str):
        if self.copy_root_to_tmp:
            assert self.tmp_dir is not None
            root = self.tmp_dir
        else:
            root = self.root_dir
        if self.basis_table is None:
            # Read the basis from the basis files provided.
            assert self.basis_files is not None
            self.basis_table = AtomicTableWithEdges.from_basis_glob(
                Path(root).glob(self.basis_files), no_basis_atoms=self.no_basis
            )

        # Initialize the data.
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        # Load json file with paths for each split
        if self.runs_json is not None:
            json_path = Path(self.runs_json)
            if not json_path.is_absolute():
                json_path = Path(root) / json_path
            with open(json_path, "r") as f:
                runs_dict = json.load(f)
        else:
            runs_dict = {}

        self.data_processor = MatrixDataProcessor(
            basis_table=self.basis_table,
            out_matrix=self.out_matrix,
            symmetric_matrix=self.symmetric_matrix,
            sub_point_matrix=self.sub_point_matrix,
            node_attr_getters=self.initial_node_feats,
        )

        # Set the paths for each split
        for split in ["train", "val", "test", "predict"]:
            glob_path = getattr(self, "%s_runs" % split)
            # Use the glob paths if given
            if glob_path is not None:
                runs = Path(root).glob(glob_path)
            # Else use the json file
            elif split in runs_dict:
                runs = [Path(root) / p for p in runs_dict[split]]
            else:
                runs = None

            if runs is not None:
                # Contruct the dataset
                # For predictions, we don't need to load the labels (actually we don't have them)
                # For the other splits, we need to load the labels (target matrices)
                dataset = TorchBasisMatrixDataset(
                    list(runs),
                    data_processor=self.data_processor,
                    data_cls=TorchBasisMatrixData,
                    load_labels=split != "predict",
                )

                if self.store_in_memory:
                    if self.rotating_pool_size and split == "train":
                        logging.warning(
                            "Does not load training data to memory because rotating_pool_size is set"
                        )
                    else:
                        logging.debug("Loading dataset split=%s into memory" % split)
                        dataset = InMemoryData(dataset)

                setattr(self, "%s_dataset" % split, dataset)

        # Now check if we should take some data out of the training data to use it for
        # validation.
        if self.val_dataset is None and self.train_dataset is not None:
            # Instantiate a random state to sample from data
            rng = np.random.RandomState(32)
            # User didn't specify a validation directory, just randomly draw 10 percent from the training.
            perms = rng.permutation(len(self.train_dataset))
            num_val = int(math.ceil(len(self.train_dataset) / 10))
            val_indices = np.sort(perms[0:num_val]).tolist()
            train_indices = np.sort(perms[num_val:]).tolist()

            self.val_dataset = torch.utils.data.Subset(self.train_dataset, val_indices)
            self.train_dataset = torch.utils.data.Subset(
                self.train_dataset, train_indices
            )

        # Wrap training dataset in rotating pool
        if self.rotating_pool_size:
            self.train_dataset = RotatingPoolData(
                self.train_dataset, self.rotating_pool_size
            )

    def train_dataloader(self):
        assert (
            self.train_dataset is not None
        ), "No training data was provided, please set the ``train_runs`` argument."
        if self.rotating_pool_size:
            assert isinstance(self.train_dataset, RotatingPoolData)
            data_handle = self.train_dataset.get_data_pool()
        else:
            data_handle = self.train_dataset
        return DataLoader(
            data_handle,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.loader_threads,
            persistent_workers=bool(self.hparams.loader_threads),
        )

    def val_dataloader(self):
        assert (
            self.val_dataset is not None
        ), "No validation data was provided, please set either the ``train_runs`` or the ``val_runs`` argument."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.loader_threads,
        )

    def test_dataloader(self):
        assert (
            self.test_dataset is not None
        ), "No test data was provided, please set the ``test_runs`` argument."
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.loader_threads,
        )

    def predict_dataloader(self):
        assert (
            self.predict_dataset is not None
        ), "No prediction data was provided, please set the ``predict_structs`` argument."
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.loader_threads,
        )
