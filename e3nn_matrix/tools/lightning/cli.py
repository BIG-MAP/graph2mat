from typing import Dict, Set, List, Any
import copy
import os
from pytorch_lightning.cli import LightningCLI, SaveConfigCallback, LightningArgumentParser
import torch
from jsonargparse import Namespace

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
        parser.add_argument("--multiprocessing_sharing_strategy", default="", type=str)

        parser.set_defaults(defaults)

    def before_instantiate_classes(self) -> None:
        # This is executed after config/argparser has been instanced
        # but before data and model has been instantiated.

        # Pytorch_lightning has this strange behavior that it can not resume
        # simply from a checkpoint file because it doesn't use the hyperparameters
        # of the model to instantiate the model. Instead, it needs you to pass
        # exactly the same config that you used when you created the model. This
        # is really weird (at least for our case). So what we do here is to put all
        # the hyperparameters into the config to "trick" lightning into loading
        # the model and the data processors from the checkpoint.
        self._load_from_checkpoint()

        import torch.multiprocessing
        config_ns = getattr(self.config, self.config.subcommand)
        if config_ns.multiprocessing_sharing_strategy:
            assert config_ns.multiprocessing_sharing_strategy in torch.multiprocessing.get_all_sharing_strategies()
            torch.multiprocessing.set_sharing_strategy(config_ns.multiprocessing_sharing_strategy)

    def _load_from_checkpoint(self):
        # Check if ckpt_path is given in subcommand
        subcommand = self.config.subcommand
        # Get namespace for current subcommand
        config_ns = getattr(self.config, subcommand)
        # Get the path of the checkpoint
        ckpt_path = getattr(config_ns, "ckpt_path", None)
        if ckpt_path:
            # Load parameters from the checkpoint file.
            # We only need the z_table data and other hyperparameters, which contain numpy arrays, but
            # the checkpoint contains the whole model, with torch tensors. The torch tensors might be located
            # on GPU (or any other device), which is possibly not available when we load the checkpoint. Map
            # those tensors to CPU so that there are no loading errors.
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

            if os.environ.get("E3MAT_FROMCKPT_DATAPROC", "").lower() not in ["off", "false", "f", "no", "0"]:
                # Extract the keys of the data module that control how to process the data.
                # And set them explicitly in the config.
                for k in ("out_matrix", "sub_atomic_matrix", "symmetric_matrix", "basis_files"):
                    config_ns.data[k] = checkpoint["datamodule_hyper_parameters"][k]

            if os.environ.get("E3MAT_FROMCKPT_MODEL", "").lower() not in ["off", "false", "f", "no", "0"]:
                # Extract the parameters that where used to instantiate the model in the first
                # place and set them in the config.
                config_ns['model'] = config_ns.__class__(checkpoint['hyper_parameters'])

            if os.environ.get("E3MAT_FROMCKPT_ZTABLE", "").lower() not in ["off", "false", "f", "no", "0"]:
                z_table = checkpoint.get("z_table")
                if z_table:
                    config_ns.data.z_table = z_table
                    config_ns.model.z_table = z_table

class SaveConfigSkipZTableCallback(SaveConfigCallback):
    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        # Make a shallow copy of config and overwrite z_table argument
        # to not save z_table to yaml which makes no sense
        config = copy.copy(config)
        if hasattr(config.data, "z_table"):
            config.data.z_table = None
            config.model.z_table = None
        super().__init__(
            parser=parser,
            config=config,
            config_filename=config_filename,
            overwrite=overwrite,
            multifile=multifile,
        )

