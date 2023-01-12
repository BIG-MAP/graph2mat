import os
from pytorch_lightning.cli import LightningCLI

class OrbitalMatrixCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.root_dir", "model.root_dir")
        parser.link_arguments("data.basis_files", "model.basis_files")
        parser.link_arguments("data.z_table", "model.z_table")
        parser.link_arguments("data.symmetric_matrix", "model.symmetric_matrix")

        # The following is a hack to set logger defaults based on environment variables
        init_args = {
                "save_dir": os.environ.get("PL_LOGGER_SAVE_DIR", "."),
                "name": os.environ.get("PL_LOGGER_NAME", "lightning_logs"),
                "version": os.environ.get("PL_LOGGER_VERSION"),
        }

        if os.environ.get("PL_LOGGER_DEFAULTS_ENABLE") is not None:
            default_loggers = [{
                "class_path": "TensorBoardLogger",
                "init_args": init_args
                },
                {
                "class_path": "CSVLogger",
                "init_args": init_args
                },
            ]

            parser.set_defaults({"trainer.logger": default_loggers})
