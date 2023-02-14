from .model import LitOrbitalMatrixMACE

from e3nn_matrix.scripts.lit import OrbitalMatrixCLI, MatrixDataModule, MatrixTrainer, SaveConfigSkipZTableCallback

def cli():
    OrbitalMatrixCLI(LitOrbitalMatrixMACE, MatrixDataModule, trainer_class=MatrixTrainer, save_config_callback=SaveConfigSkipZTableCallback)
