import sisl
import numpy as np

from dataclasses import dataclass

from ...table import AtomicTableWithEdges
from .orbital_matrix import OrbitalMatrix


@dataclass
class DensityMatrix(OrbitalMatrix):
    def get_atomic_matrices(self, z_table: AtomicTableWithEdges):
        return z_table.atomic_DM


def get_atomic_DM(atom: sisl.Atom) -> np.ndarray:
    """Gets the block corresponding to the atomic density matrix.

    Parameters
    ----------
    atom: sisl.Atom
        The Atom object from which the density block is desired.
        It must contain the basis orbitals, with the initial occupation for each of them.
        This is how they come if you have read the basis from a SIESTA calculation or
        from an .ion file.

    Returns
    ----------
    np.ndarray of shape atom.no x atom.no
        Square matrix encoding the isolated atom density matrix.
    """
    return np.diag(atom.q0)
