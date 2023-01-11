from typing import Optional, Union, Literal

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import sisl

from .matrices import OrbitalMatrix, get_matrix_cls
from .sparse import csr_to_block_dict

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)
PhysicsMatrixType = Literal["density_matrix", "hamiltonian", "energy_density_matrix", "dynamical_matrix"]

DEFAULT_CONFIG_TYPE = "Default"

@dataclass
class OrbitalConfiguration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    atoms: sisl.Atoms
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None
    matrix: Optional[OrbitalMatrix] = None

    weight: float = 1.0  # weight of config in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config

    @classmethod
    def from_geometry(cls, geometry: sisl.Geometry, **kwargs) -> "OrbitalConfiguration":

        if "pbc" not in kwargs:
            kwargs['pbc'] = (True, True, True)

        return cls(atomic_numbers=geometry.atoms.Z, atoms=geometry.atoms, positions=geometry.xyz, cell=geometry.cell, **kwargs)


def load_orbital_config_from_run(
    runfilepath: Union[str, Path], 
    out_matrix: Optional[PhysicsMatrixType] = None
) -> OrbitalConfiguration:
    """Initializes an OrbitalConfiguration object from the main input file of a run.
    Parameters
    -----------
    runfilepath: str or Path
        The path of the main input file. E.g. in SIESTA this is the path to the ".fdf" file
    out_matrix: {"density_matrix", "hamiltonian", "energy_density_matrix", "dynamical_matrix", None}
        The matrix to be read from the output of the run. The configuration object will
        contain the matrix.
        If it is None, then no matrices are read from the output. This is the case when trying to
        predict matrices, since you don't have the output yet.
    """
    # Initialize the file object for the main input file
    main_input = sisl.get_sile(runfilepath)

    if out_matrix is not None:
        # Get the method to read the desired matrix and read it
        read = getattr(main_input, f"read_{out_matrix}")
        matrix = read()

        # Determine the dataclass that should store the matrix and build the block dict
        # sparse structure.
        matrix_cls = get_matrix_cls(out_matrix)
        matrix_block = csr_to_block_dict(matrix._csr, matrix.atoms, nsc=matrix.nsc, matrix_cls=matrix_cls)

        # The matrix will have an associated geometry, so we will use it.
        geometry = matrix.geometry
    else:
        # We have no matrix to read, we will just read the geometry.
        geometry = main_input.read_geometry()
        matrix_block = None

    return OrbitalConfiguration.from_geometry(geometry=geometry, matrix=matrix_block)
