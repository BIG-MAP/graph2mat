from pytorch_lightning.cli import LightningCLI

class OrbitalMatrixCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.root_dir", "model.root_dir")
        parser.link_arguments("data.basis_files", "model.basis_files")
        parser.link_arguments("data.z_table", "model.z_table")
        parser.link_arguments("data.symmetric_matrix", "model.symmetric_matrix")