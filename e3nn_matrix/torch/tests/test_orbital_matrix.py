"""Tests for input preparation"""
from e3nn_matrix.data.sparse import csr_to_block_dict
from e3nn_matrix.data.configuration import OrbitalConfiguration
from e3nn_matrix.torch.data import OrbitalMatrixData
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges

def test_orbital_matrix_data(density_matrix, density_z_table):
    # For now we just test that we can get an OrbitalMatrixData object
    # with and without matrix, and nothing breaks.

    block_dict = csr_to_block_dict(density_matrix._csr, density_matrix.atoms, nsc=density_matrix.nsc)
    config = OrbitalConfiguration.from_geometry(geometry=density_matrix.geometry, matrix=block_dict)
    data = OrbitalMatrixData.from_config(config, z_table=density_z_table)

    no_matrix_config = OrbitalConfiguration.from_geometry(geometry=density_matrix.geometry, matrix=block_dict)
    no_matrix_data = OrbitalMatrixData.from_config(config, z_table=density_z_table)


