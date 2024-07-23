"""Implements a custom CLI that slightly tweaks pytorch_lightning's default.

Most
"""

import copy
import os
from pytorch_lightning.cli import (
    LightningCLI,
    SaveConfigCallback,
    LightningArgumentParser,
    ArgsType,
)
import torch
from jsonargparse import Namespace
from jsonargparse._typehints import ActionTypeHint

from graph2mat.bindings.torch.load import sanitize_checkpoint


class OrbitalMatrixCLI(LightningCLI):
    """Custom pytorch_lightning CLI optimized for matrix learning.

    There are some defaults that change.

    However, the most relevant change is that when loading a checkpoint
    with the ``ckpt_path`` key, all the options stored in the checkpoint
    file will be used as defaults. This change was made so that you can
    just load a checkpoint file and use it without needing to provide
    all the settings that were used to generate that checkpoint (which
    is the way raw pytorch_lightning works).
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments("data.root_dir", "model.root_dir")
        parser.link_arguments("data.basis_files", "model.basis_files")
        parser.link_arguments("data.basis_table", "model.basis_table")
        parser.link_arguments("data.no_basis", "model.no_basis")
        parser.link_arguments("data.symmetric_matrix", "model.symmetric_matrix")
        parser.link_arguments("data.initial_node_feats", "model.initial_node_feats")

        defaults = {}
        # Set logger defaults based on environment variables
        if os.environ.get("PL_LOGGER_DEFAULTS_ENABLE") is not None:
            init_args = {
                "save_dir": os.environ.get("PL_LOGGER_SAVE_DIR", "."),
                "name": os.environ.get("PL_LOGGER_NAME", "lightning_logs"),
                "version": os.environ.get("PL_LOGGER_VERSION"),
            }

            default_loggers = [
                {"class_path": "TensorBoardLogger", "init_args": init_args},
                {"class_path": "CSVLogger", "init_args": init_args},
            ]
            defaults["trainer.logger"] = default_loggers

        # saves last checkpoints based on "step" metric
        # as well as "val_loss" metric
        init_args_last = {
            "monitor": "step",
            "mode": "max",
            "filename": "last-{step:02d}",
            "save_last": True,
            "auto_insert_metric_name": False,
        }
        init_args_best = {
            "monitor": "val_loss",
            "mode": "min",
            "filename": "best-{step:02d}",
            "auto_insert_metric_name": False,
        }
        default_callbacks = [
            {
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

    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        """Parses command line arguments and stores it in ``self.config``."""

        super().parse_arguments(parser, args)

        # Pytorch_lightning has this strange behavior that it can not resume
        # simply from a checkpoint file because it doesn't use the hyperparameters
        # of the model to instantiate the model. Instead, it needs you to pass
        # exactly the same config that you used when you created the model. This
        # is really weird (at least for our case). So what we do here is: if the user
        # has provided a checkpoint file, use all the parameters of the checkpoint file
        # as defaults.

        # Check if ckpt_path is given in subcommand
        subcommand = self.config.subcommand
        # Get namespace for current subcommand
        config_ns = getattr(self.config, subcommand)
        # Get the path of the checkpoint
        ckpt_path = getattr(config_ns, "ckpt_path", None)

        # If there is a checkpoint path, we need to reparse the arguments, using the
        # checkpoint parameters as defaults.
        if ckpt_path is not None:
            defaults = self._config_from_ckpt(ckpt_path)

            subcommand_parser = self._parser(subcommand)

            # Arguments are defined as "k.subkey" in lightning. E.g. model.num_neighbours
            # So we need to convert the dict to those keys. Another problem is that some
            # arguments are linked and therefore "data.x" might not exist because it is linked
            # to "model.x". That's why we need to set the defaults one by one inside a try/except
            # block (I found no way to check if the argument is defined in the parser).
            with ActionTypeHint.allow_default_instance_context():
                for k in defaults:
                    for subkey in defaults[k]:
                        try:
                            subcommand_parser.set_defaults(
                                {f"{k}.{subkey}": defaults[k][subkey]}
                            )
                        except KeyError:
                            pass

            # We have set all the right defaults now! So we can reparse the arguments.
            super().parse_arguments(parser, args)

    def before_instantiate_classes(self) -> None:
        # This is executed after config/argparser has been instanced
        # but before data and model has been instantiated.
        import torch.multiprocessing

        config_ns = getattr(self.config, self.config.subcommand)
        if config_ns.multiprocessing_sharing_strategy:
            assert (
                config_ns.multiprocessing_sharing_strategy
                in torch.multiprocessing.get_all_sharing_strategies()
            )
            torch.multiprocessing.set_sharing_strategy(
                config_ns.multiprocessing_sharing_strategy
            )

    def _config_from_ckpt(self, ckpt_path: str):
        # Load parameters from the checkpoint file.
        # We only need to load hyperparameters, not weights. Therefore, we don't care whether things were in
        # the GPU. The torch tensors might be located on GPU (or any other device), which is possibly not
        # available when we load the checkpoint. Map those tensors to CPU so that there are no loading errors.
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        checkpoint = sanitize_checkpoint(checkpoint)

        config = {}

        if os.environ.get("E3MAT_FROMCKPT_DATAPROC", "").lower() not in [
            "off",
            "false",
            "f",
            "no",
            "0",
        ]:
            # Extract the keys of the data module that control how to process the data.
            # And set them explicitly in the config.
            config["data"] = checkpoint["datamodule_hyper_parameters"]

        if os.environ.get("E3MAT_FROMCKPT_MODEL", "").lower() not in [
            "off",
            "false",
            "f",
            "no",
            "0",
        ]:
            # Extract the parameters that where used to instantiate the model in the first
            # place and set them in the config.
            config["model"] = checkpoint["hyper_parameters"]

        if os.environ.get("E3MAT_FROMCKPT_BASISTABLE", "").lower() not in [
            "off",
            "false",
            "f",
            "no",
            "0",
        ]:
            basis_table = checkpoint.get("basis_table")
            if basis_table:
                config["model"]["basis_table"] = basis_table
                config["data"]["basis_table"] = basis_table

        return config


class SaveConfigSkipBasisTableCallback(SaveConfigCallback):
    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
    ) -> None:
        # Make a shallow copy of config and overwrite basis_table argument
        # to not save basis_table to yaml which makes no sense
        config = copy.copy(config)
        if hasattr(config.data, "basis_table"):
            config.data.basis_table = None
            config.model.basis_table = None
        super().__init__(
            parser=parser,
            config=config,
            config_filename=config_filename,
            overwrite=overwrite,
            multifile=multifile,
        )
