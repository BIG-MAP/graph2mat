import os
from pytorch_lightning.cli import LightningCLI

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
