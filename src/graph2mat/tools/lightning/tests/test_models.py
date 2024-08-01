"""This file tests functionality that ALL models should have.

This includes for example input-output equivariance and the
ability to deal with periodic boundary conditions.
"""
# import numpy as np
# import sisl
# import torch

# from scipy.spatial.transform import Rotation

# from graph2mat.data.processing import MatrixDataProcessor
# from graph2mat.data.configuration import OrbitalConfiguration
# from graph2mat.torch.data import BasisMatrixTorchData
# from graph2mat.data.table import AtomicTableWithEdges
# from graph2mat.data.irreps_tools import get_atom_irreps
# from graph2mat.data.sparse import (
#     nodes_and_edges_to_coo,
#     nodes_and_edges_to_sparse_orbital,
# )

# from graph2mat.tools.lightning.models.mace import LitOrbitalMatrixMACE

# import pytest


# s_orbitals = [sisl.AtomicOrbital(n=1, l=0, R=2.5)]
# p_orbitals = [
#     sisl.AtomicOrbital(n=2, l=1, m=-1, R=3),
#     sisl.AtomicOrbital(n=2, l=1, m=0, R=3),
#     sisl.AtomicOrbital(n=2, l=1, m=1, R=3),
# ]
# d_orbitals = [
#     sisl.AtomicOrbital(n=2, l=2, m=-2, R=3),
#     sisl.AtomicOrbital(n=2, l=2, m=-1, R=3),
#     sisl.AtomicOrbital(n=2, l=2, m=0, R=3),
#     sisl.AtomicOrbital(n=2, l=2, m=1, R=3),
#     sisl.AtomicOrbital(n=2, l=2, m=2, R=3),
# ]


# @pytest.fixture(scope="module", params=("same_basis", "DZP"))
# def basis_shape(request):
#     return request.param


# @pytest.fixture(scope="module")
# def z_table(basis_shape):
#     if basis_shape == "same_basis":
#         H = sisl.Atom("H", orbitals=[*s_orbitals, *p_orbitals])
#         O = sisl.Atom("O", orbitals=[*s_orbitals, *p_orbitals])
#     elif basis_shape == "DZP":
#         H = sisl.Atom(
#             "H", orbitals=[*s_orbitals, *s_orbitals, *p_orbitals, *p_orbitals]
#         )
#         O = sisl.Atom(
#             "O",
#             orbitals=[*s_orbitals, *s_orbitals, *p_orbitals, *p_orbitals, *d_orbitals],
#         )
#     else:
#         raise ValueError("basis_shape was provided a wrong value")

#     return AtomicTableWithEdges([H, O])


# @pytest.fixture(scope="module")
# def data_processor(z_table):
#     return MatrixDataProcessor(symmetric_matrix=True, basis_table=z_table)


# @pytest.fixture(scope="module", params=["mace"])
# def model(
#     data_processor,
#     request,
# ):
#     model_name = request.param

#     if model_name == "mace":
#         return LitOrbitalMatrixMACE(
#             basis_table=data_processor.basis_table,
#             symmetric_matrix=data_processor.symmetric_matrix,
#             avg_num_neighbors=1,
#             correlation=2,
#             max_ell=2,
#             num_interactions=2,
#             hidden_irreps="10x0e + 10x1o + 10x2e",
#             edge_hidden_irreps="4x0e + 4x1o + 4x2e",
#         )


# @pytest.fixture(scope="module", params=("bimolec", "square", "chain"))
# def geometry(request, z_table):
#     [H, O] = z_table.atoms
#     system = request.param
#     if system == "bimolec":
#         geom = sisl.Geometry([[0, 0, 0], [1, 0, 0]], atoms=[H, O], lattice=[20, 20, 20])
#     elif system == "square":
#         geom = sisl.Geometry(
#             [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
#             atoms=[H, O, H, O],
#             lattice=[20, 20, 20],
#         )
#     elif system == "chain":
#         geom = sisl.Geometry([[0, 0, 0], [2, 0, 0]], atoms=[H, O], lattice=[4, 20, 20])
#         geom.set_nsc([3, 1, 1])
#     else:
#         raise ValueError("system was provided a wrong value")

#     return geom


# def test_model_equivariance(model, data_processor, geometry):
#     geom = geometry
#     z_table = data_processor.basis_table

#     # Helper function that returns the predicted matrix from a geometry.
#     def get_matrix(geom):
#         config = OrbitalConfiguration.from_geometry(geom)
#         data = BasisMatrixTorchData.from_config(config, data_processor=data_processor)

#         out = model.model(data)

#         return nodes_and_edges_to_coo(
#             out["node_labels"].detach().numpy(),
#             z_table.atom_block_pointer(data.point_types),
#             out["edge_labels"].detach().numpy(),
#             data.edge_index[:, ::2].numpy(),
#             z_table.edge_block_pointer(data.edge_types[::2]),
#             geom.orbitals,
#             n_supercells=geometry.n_s,
#             edge_neigh_isc=data.neigh_isc[::2],
#             symmetrize_edges=True,
#         ).toarray()

#     atom_irreps = [get_atom_irreps(atom) for atom in z_table.atoms]
#     irreps = None
#     for at in geom.atoms.specie:
#         if irreps is None:
#             irreps = atom_irreps[at]
#         else:
#             irreps = irreps + atom_irreps[at]
#     # And the rotated one
#     if geometry.n_s > 1:
#         # If it's a periodic structure we just rotate by 90 degrees around the z axis
#         # so that we can easily tell what the new required supercell will be, and
#         # also so that we can very easily compare the matrices.
#         assert np.all(
#             geometry.cell[:2, 2] == 0
#         ), "Testing the equivariance for periodic cells where the first or second cell has a non-zero z component is not supported."
#         R = Rotation.from_rotvec(np.pi / 2 * np.array([0, 0, 1])).as_matrix()
#     else:
#         R = Rotation.from_euler("xyz", [20, 30, 50]).as_matrix()

#     rot_geom = geom.copy()
#     rot_geom.xyz = rot_geom.xyz @ R.T
#     rot_geom.cell[:] = rot_geom.cell @ R.T
#     if geometry.n_s > 1:
#         rot_geom.set_nsc([geometry.nsc[1], geometry.nsc[0], geometry.nsc[2]])

#     # Get the predicted matrices for both
#     out = get_matrix(geom)
#     rot_out = get_matrix(rot_geom)

#     # Get the matrix that rotates the unrotated geometry output to get the expected output.
#     # Note that a change of basis to the spherical harmonics is needed.
#     basis_change = data_processor.basis_table.change_of_basis
#     D = irreps.D_from_matrix(
#         torch.tensor(basis_change @ R @ basis_change.T, dtype=torch.get_default_dtype())
#     )

#     for isc in range(geometry.n_s):
#         cell_out = out[:, geometry.no * isc : geometry.no * (isc + 1)]
#         cell_rot_out = rot_out[:, geometry.no * isc : geometry.no * (isc + 1)]
#         # Check that the expected rotated output is the same as the produced output.
#         # We set the tolerance to 50 pico, which is quite high, but it is the precision that we have seen to acheive with
#         # float32. With float64 you can go higher.
#         assert np.allclose(D @ cell_out @ D.T, cell_rot_out, atol=5e-5), isc


# def test_model_supercell(model, data_processor, geometry):
#     """Checks if tiling the output of the model gives the same result as tiling the input.

#     A periodic structure should produce exactly the same output regardless of the size of the supercell,
#     and that is what we test here.
#     """
#     geom = geometry
#     z_table = data_processor.basis_table

#     # Helper function that returns the predicted matrix from a geometry.
#     def get_matrix(geom):
#         config = OrbitalConfiguration.from_geometry(geom)
#         data = BasisMatrixTorchData.from_config(config, data_processor=data_processor)

#         out = model.model(data)

#         return nodes_and_edges_to_sparse_orbital(
#             out["node_labels"].detach().numpy(),
#             z_table.atom_block_pointer(data.point_types),
#             out["edge_labels"].detach().numpy(),
#             data.edge_index[:, ::2].numpy(),
#             z_table.edge_block_pointer(data.edge_types[::2]),
#             geom,
#             edge_neigh_isc=data.neigh_isc[::2],
#             symmetrize_edges=True,
#         )

#     # Get the predicted matrices for both
#     out = get_matrix(geom)
#     out_array = out.tile(2, 0).tile(2, 1).tile(2, 2)._csr.todense()

#     tiled_out = get_matrix(geom.tile(2, 0).tile(2, 1).tile(2, 2))
#     tiled_out_array = tiled_out._csr.todense()

#     assert np.allclose(out_array, tiled_out_array, atol=1e-6), np.max(
#         np.abs(out_array - tiled_out_array)
#     )
