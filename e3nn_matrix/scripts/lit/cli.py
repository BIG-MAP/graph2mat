from typing import Dict, Set
import os
from pytorch_lightning.cli import LightningCLI
import torch

class OrbitalMatrixCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.root_dir", "model.root_dir")
        parser.link_arguments("data.basis_files", "model.basis_files")
        parser.link_arguments("data.z_table", "model.z_table")
        parser.link_arguments("data.symmetric_matrix", "model.symmetric_matrix")

        defaults = {}
        # Set logger defaults based on environment variables
        if os.environ.get("PL_LOGGER_DEFAULTS_ENABLE") is not None:
            init_args = {
                    "save_dir": os.environ.get("PL_LOGGER_SAVE_DIR", "."),
                    "name": os.environ.get("PL_LOGGER_NAME", "lightning_logs"),
                    "version": os.environ.get("PL_LOGGER_VERSION"),
            }

            default_loggers = [{
                "class_path": "TensorBoardLogger",
                "init_args": init_args
                },
                {
                "class_path": "CSVLogger",
                "init_args": init_args
                },
            ]
            defaults["trainer.logger"] = default_loggers


        # saves last checkpoints based on "step" metric
        # as well as "val_loss" metric
        init_args_last = {
            "monitor":"step",
            "mode":"max",
            "filename":"last-{step:02d}",
            "save_last": True,
            "auto_insert_metric_name": False,
        }
        init_args_best = {
            "monitor":"val_loss",
            "mode":"min",
            "filename":"best-{step:02d}",
            "auto_insert_metric_name": False,
        }
        default_callbacks = [{
            "class_path": "ModelCheckpoint",
            "init_args": init_args_last,
            },
            {
            "class_path": "ModelCheckpoint",
            "init_args": init_args_best,
            },
        ]
        defaults["trainer.callbacks"] = default_callbacks

        parser.set_defaults(defaults)

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        # TODO: Not clear what "to skip" means?
        subcmd_dict = super(OrbitalMatrixCLI, OrbitalMatrixCLI).subcommands()
        subcmd_dict["serve"] = {"model", "datamodule"}
        return subcmd_dict

    def before_instantiate_classes(self) -> None:
        # This is executed after config/argparser has been instanced
        # but before data and model has been instantiated.

        # The data module can not load the z_table from a checkpoint
        # because when checkpoint loading happens, the data loaders
        # might already be instanced.
        # Therefore we try to load the z_table from checkpoint and
        # put it in the config as an object before the data module is loaded.
        self._load_z_table_from_checkpoint()

    def _load_z_table_from_checkpoint(self):
        # Check if ckpt_path is given in subcommand
        subcommand = self.config.subcommand
        # Get namespace for current subcommand
        config_ns = getattr(self.config, subcommand)
        # Get the path of the checkpoint
        ckpt_path = getattr(config_ns, "ckpt_path", None)
        if ckpt_path:
            # Load the z_table and assign to model and data
            checkpoint = torch.load(ckpt_path)
            z_table = checkpoint.get("z_table")
            if z_table:
                config_ns.data.z_table = z_table
                config_ns.model.z_table = z_table
