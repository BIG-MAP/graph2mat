# """Tests for input preparation"""
# from graph2mat.data.sparse import csr_to_block_dict
# from graph2mat.data.configuration import OrbitalConfiguration
# from graph2mat.torch.data import BasisMatrixTorchData


# def test_orbital_matrix_data(density_matrix, density_data_processor):
#     # For now we just test that we can get an OrbitalMatrixData object
#     # with and without matrix, and nothing breaks.

#     block_dict = csr_to_block_dict(
#         density_matrix._csr, density_matrix.atoms, nsc=density_matrix.nsc
#     )
#     config = OrbitalConfiguration.from_geometry(
#         geometry=density_matrix.geometry, matrix=block_dict
#     )
#     data = BasisMatrixTorchData.from_config(
#         config, data_processor=density_data_processor
#     )

#     no_matrix_config = OrbitalConfiguration.from_geometry(
#         geometry=density_matrix.geometry, matrix=block_dict
#     )
#     no_matrix_data = BasisMatrixTorchData.from_config(
#         config, data_processor=density_data_processor
#     )
