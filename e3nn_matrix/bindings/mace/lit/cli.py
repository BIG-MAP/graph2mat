from .model import LitOrbitalMatrixMACE

from e3nn_matrix.scripts.lit import OrbitalMatrixCLI, MatrixDataModule, MatrixTrainer

def cli():
    OrbitalMatrixCLI(LitOrbitalMatrixMACE, MatrixDataModule, trainer_class=MatrixTrainer)
