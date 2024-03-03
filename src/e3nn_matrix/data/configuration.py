"""Implements classes to store an example of the dataset in memory.

A "configuration" is an object that contains all the information about a
given example in the dataset. It contains all the features needed to
describe the example (e.g. coordinates, lattice vectors...), and optionally
the matrix that corresponds to this example.

In a typical case, your configurations will contain the matrix as a label
for training, validating or testing. When doing inference, the configurations
will not have an associated matrix, since the matrix is what you are trying
to calculate.
"""

from typing import Optional, Union, Literal, Dict, Any, Sequence

import warnings

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import sisl

from .basis import PointBasis
from .matrices import OrbitalMatrix, BasisMatrix, get_matrix_cls
from .sparse import csr_to_block_dict

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)
PhysicsMatrixType = Literal[
    "density_matrix", "hamiltonian", "energy_density_matrix", "dynamical_matrix"
]

DEFAULT_CONFIG_TYPE = "Default"


@dataclass
class BasisConfiguration:
    """Stores a distribution of points in space, with associated basis functions.

    Optionally, it can also store an associated matrix.

    Parameters
    -----------
    point_types:
        Shape (n_points,).
        The type of each point. Each type can be either a string or an integer,
        and it should be the type key of a `PointBasis` object in the `basis` list.
    positions:
        Shape (n_points, 3).
        The positions of each point in cartesian coordinates.
    basis:
        List of `PointBasis` objects for types that are (possibly) present in the system.
    cell:
        Shape (3, 3).
        The cell vectors that delimit the system, in cartesian coordinates.
    pbc:
        Shape (3,).
        Whether the system is periodic in each cell direction.
    matrix:
        The matrix associated to the configuration.
    weight:
        The weight of the configuration in the loss.
    config_type:
        A string that indicates the type of configuration.
    metadata:
        A dictionary with additional metadata related to the configuration.
    """

    point_types: np.ndarray
    positions: Positions
    basis: Sequence[PointBasis]
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None
    matrix: Optional[BasisMatrix] = None

    weight: float = 1.0  # weight of config in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config
    metadata: Optional[Dict[str, Any]] = None

    def to_sisl_geometry(self) -> sisl.Geometry:
        """Converts the configuration to a sisl Geometry."""

        atoms = {pb.type: pb.to_sisl_atom(Z=i + 1) for i, pb in enumerate(self.basis)}

        return sisl.Geometry(
            xyz=self.positions,
            atoms=[atoms[k] for k in self.point_types],
            lattice=self.cell,
        )


@dataclass
class OrbitalConfiguration(BasisConfiguration):
    """Stores a distribution of atoms in space, with associated orbitals.

    Optionally, it can also store an associated matrix.

    This is a version of `BasisConfiguration` for atomic systems,
    where points are atoms.

    Parameters
    -----------
    point_types:
        Shape (n_points,).
        The type of each point. Each type can be either a string or an integer,
        and it should be the type key of a `PointBasis` object in the `basis` list.
    positions:
        Shape (n_points, 3).
        The positions of each point in cartesian coordinates.
    basis:
        Atoms that are (possibly) present in the system.
    cell:
        Shape (3, 3).
        The cell vectors that delimit the system, in cartesian coordinates.
    pbc:
        Shape (3,).
        Whether the system is periodic in each cell direction.
    matrix:
        The matrix associated to the configuration.
    weight:
        The weight of the configuration in the loss.
    config_type:
        A string that indicates the type of configuration.
    metadata:
        A dictionary with additional metadata related to the configuration.
    """

    point_types: np.ndarray
    positions: Positions  # Angstrom
    basis: sisl.Atoms
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None
    matrix: Optional[OrbitalMatrix] = None

    weight: float = 1.0  # weight of config in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config
    metadata: Optional[Dict[str, Any]] = None

    @property
    def atom_types(self) -> np.ndarray:
        return self.point_types

    @property
    def atoms(self) -> sisl.Atoms:
        return self.basis

    @classmethod
    def new(
        cls,
        obj: Union[sisl.Geometry, sisl.SparseOrbital, str, Path],
        labels: bool = True,
        **kwargs,
    ) -> "OrbitalConfiguration":
        if isinstance(obj, sisl.Geometry):
            if labels:
                raise ValueError(
                    "Cannot infer output labels only from a geometry. Please provide either a matrix or a path to a run file."
                )
            return cls.from_geometry(obj, **kwargs)
        elif isinstance(obj, sisl.SparseOrbital):
            return cls.from_matrix(obj, labels=labels, **kwargs)
        elif isinstance(obj, (str, Path)):
            if not labels:
                kwargs["out_matrix"] = None
            return cls.from_run(obj, **kwargs)
        else:
            raise TypeError(
                f"Cannot create OrbitalConfiguration from {obj.__class__.__name__}."
            )

    @classmethod
    def from_geometry(cls, geometry: sisl.Geometry, **kwargs) -> "OrbitalConfiguration":
        """Initializes an OrbitalConfiguration object from a sisl geometry.

        Note that the created object will not have an associated matrix, unless it is passed
        explicitly as a keyword argument.

        Parameters
        -----------
        geometry: sisl.Geometry
            The geometry to associate to the OrbitalConfiguration.
        **kwargs:
            Additional arguments to be passed to the OrbitalConfiguration constructor.
        """

        if "pbc" not in kwargs:
            kwargs["pbc"] = (True, True, True)

        return cls(
            point_types=geometry.atoms.Z,
            basis=geometry.atoms,
            positions=geometry.xyz,
            cell=geometry.cell,
            **kwargs,
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: sisl.SparseOrbital,
        geometry: Union[sisl.Geometry, None] = None,
        labels: bool = True,
        **kwargs,
    ) -> "OrbitalConfiguration":
        """Initializes an OrbitalConfiguration object from a sisl matrix.

        Parameters
        -----------
        matrix: sisl.SparseOrbital
            The matrix to associate to the OrbitalConfiguration. This matrix should have an associated
            geometry, which will be used.
        geometry: sisl.Geometry, optional
            The geometry to associate to the OrbitalConfiguration. If None, the geometry of the matrix
            will be used.
        labels: bool
            Whether to process the labels from the matrix. If False, the only thing to read
            will be the atomic structure, which is likely the input of your model.
        **kwargs:
            Additional arguments to be passed to the OrbitalConfiguration constructor.
        """
        if geometry is None:
            # The matrix will have an associated geometry, so we will use it.
            geometry = matrix.geometry

        if labels:
            # Determine the dataclass that should store the matrix and build the block dict
            # sparse structure.
            matrix_cls = get_matrix_cls(matrix.__class__)
            matrix_block = csr_to_block_dict(
                matrix._csr, matrix.atoms, nsc=matrix.nsc, matrix_cls=matrix_cls
            )

            kwargs["matrix"] = matrix_block

        return cls.from_geometry(geometry=geometry, **kwargs)

    @classmethod
    def from_run(
        cls,
        runfilepath: Union[str, Path],
        out_matrix: Optional[PhysicsMatrixType] = None,
        basis: Optional[sisl.Atoms] = None,
    ) -> "OrbitalConfiguration":
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
        # Build some metadata so that the OrbitalConfiguration object can be traced back to the run.
        metadata = {"path": runfilepath}

        def _change_geometry_basis(geometry, basis):
            # new_atoms = geometry.atoms.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for atom in [*geometry.atoms.atom]:
                    for basis_at in basis:
                        if atom.tag == basis_at.tag:
                            geometry.atoms[atom.tag] = basis_at
                            break
                    else:
                        raise ValueError(
                            f"Atom '{atom.tag}' not found in the provided basis"
                        )

        def _read_geometry(main_input, basis):
            # Read the geometry from the main input file
            try:
                geometry = main_input.read_geometry(output=True)
            except TypeError:
                geometry = main_input.read_geometry()

            if basis is not None:
                _change_geometry_basis(geometry, basis)

            return geometry

        if out_matrix is not None:
            # Get the method to read the desired matrix and read it
            read = getattr(main_input, f"read_{out_matrix}")
            if basis is not None:
                geometry = _read_geometry(main_input, basis)
                matrix = read(geometry=geometry)
            else:
                matrix = read()

            # Now build the OrbitalConfiguration object using this matrix.
            return cls.from_matrix(matrix=matrix, metadata=metadata)
        else:
            # We have no matrix to read, we will just read the geometry.
            geometry = _read_geometry(main_input, basis)

            # And build the OrbitalConfiguration object using this geometry.
            return cls.from_geometry(geometry=geometry, metadata=metadata)
