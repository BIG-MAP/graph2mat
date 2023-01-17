from .model import LitOrbitalMatrixMACE

from e3nn_matrix.scripts.lit import OrbitalMatrixCLI, MatrixDataModule

def cli():
    OrbitalMatrixCLI(LitOrbitalMatrixMACE, MatrixDataModule)