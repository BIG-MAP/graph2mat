"""Tests for sparse structure conversion"""
import pytest
import sisl
from numpy.random import RandomState
import numpy as np

from e3nn_matrix.data.sparse import csr_to_block_dict
from e3nn_matrix.data.configuration import OrbitalConfiguration
from e3nn_matrix.torch.data import OrbitalMatrixData
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges

@pytest.fixture(scope="session", params=[True, False])
def periodic(request):
    return request.param

@pytest.fixture(scope="session")
def density_matrix(periodic):

    rs = RandomState(32)

    C = sisl.Atom("C", orbitals=[
        sisl.AtomicOrbital(n=2, l=0, R=3),
        sisl.AtomicOrbital(n=2, l=1, m=-1, R=3), sisl.AtomicOrbital(n=2, l=1, m=0, R=3), sisl.AtomicOrbital(n=2, l=1, m=1, R=3)
    ])

    N = sisl.Atom("N", orbitals=[
        sisl.AtomicOrbital(n=2, l=0, R=3),
        sisl.AtomicOrbital(n=2, l=1, m=-1, R=3), sisl.AtomicOrbital(n=2, l=1, m=0, R=3), sisl.AtomicOrbital(n=2, l=1, m=1, R=3),
        #sisl.AtomicOrbital(n=2, l=1, m=-1, R=10), sisl.AtomicOrbital(n=2, l=1, m=0, R=10), sisl.AtomicOrbital(n=2, l=1, m=1, R=10),
    ])

    geom = sisl.geom.graphene_nanoribbon(width=3, atoms=[C, N])
    if not periodic:
        # Don't consider periodicity
        geom.cell[0, 0] = 20 
        geom.set_nsc([1,1,1])
    else:
        geom.set_nsc([5, 1, 1])

    dm = sisl.DensityMatrix(geom, )

    rows = dm.geometry.firsto[:-1]
    cols = np.tile(rows, dm.n_s).reshape(dm.n_s, -1) + (np.arange(dm.n_s) * dm.no).reshape(-1, 1)
    cols = cols.ravel()

    vals = (rs.random(cols.shape[0]) * 2) - 1
    for row in rows:
        dists = dm.geometry.rij(dm.o2a(row), dm.o2a(cols))
        dm[row, cols[dists < 6]] = vals[dists < 6]

    return dm

@pytest.fixture(scope="session")
def density_z_table(density_matrix):
    return AtomicTableWithEdges(density_matrix.atoms)

@pytest.fixture(scope="session")
def density_config(density_matrix):
    geometry = density_matrix.geometry

    dm_block = csr_to_block_dict(density_matrix._csr, density_matrix.atoms, nsc=density_matrix.nsc)

    return OrbitalConfiguration.from_geometry(geometry=geometry, matrix=dm_block)

@pytest.fixture(scope="session")
def density_data(density_config, density_matrix):
    z_table = AtomicTableWithEdges(density_matrix.atoms.atom)
    return OrbitalMatrixData.from_config(density_config, z_table=z_table, sub_atomic_matrix=False)