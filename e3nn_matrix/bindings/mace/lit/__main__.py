#from context import mace
from .model import LitOrbitalMatrixMACE

from e3nn_matrix.scripts.lit import OrbitalMatrixCLI, MatrixDataModule

def cli_main():
    cli = OrbitalMatrixCLI(LitOrbitalMatrixMACE, MatrixDataModule)

cli_main()