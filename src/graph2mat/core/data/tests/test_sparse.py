"""Tests for sparse structure conversion"""
import sisl
import numpy as np

from graph2mat.core.data.sparse import (
    csr_to_block_dict,
    block_dict_to_csr,
    nodes_and_edges_to_csr,
    csr_to_sisl_sparse_orbital,
    nodes_and_edges_to_sparse_orbital,
)


def test_csr_to_block_dict_simple(density_matrix):
    density_matrix = density_matrix.copy()

    density_matrix._csr.data[:] = 0
    density_matrix[0, 0] = 1
    density_matrix[0, density_matrix.orbitals[0]] = 2
    density_matrix[density_matrix.orbitals[0], 0] = 3

    block_dict = csr_to_block_dict(
        density_matrix._csr, density_matrix.atoms, nsc=density_matrix.nsc
    )

    for (i_at, j_at), val in zip([(0, 0), (0, 1), (1, 0)], [1, 2, 3]):
        assert block_dict.block_dict[i_at, j_at, 0][0, 0] == val
        assert (~np.isnan(block_dict.block_dict[i_at, j_at, 0])).sum() == 1


def test_block_dict_to_csr_simple(density_matrix):
    density_matrix = density_matrix.copy()

    first_orb_atom1 = density_matrix.orbitals[0]

    density_matrix._csr.data[:] = 0
    density_matrix[0, 0] = 1
    density_matrix[0, first_orb_atom1] = 2
    density_matrix[first_orb_atom1, 0] = 3

    block_dict = csr_to_block_dict(
        density_matrix._csr, density_matrix.atoms, nsc=density_matrix.nsc
    )

    for (i_at, j_at), val in zip([(0, 0), (0, 1), (1, 0)], [1, 2, 3]):
        assert block_dict.block_dict[i_at, j_at, 0][0, 0] == val
        assert (~np.isnan(block_dict.block_dict[i_at, j_at, 0])).sum() == 1

    new_csr = block_dict_to_csr(
        block_dict.block_dict, density_matrix.firsto, n_supercells=density_matrix.n_s
    )

    assert (new_csr.data != 0).sum() == 3
    for (i, j), val in zip(
        [(0, 0), (0, first_orb_atom1), (first_orb_atom1, 0)], [1, 2, 3]
    ):
        assert new_csr[i, j] == val


def test_full_block_dict_csr(density_matrix):
    csr = density_matrix._csr
    block_dict = csr_to_block_dict(csr, density_matrix.atoms, nsc=density_matrix.nsc)

    new_csr = block_dict_to_csr(
        block_dict.block_dict, density_matrix.firsto, n_supercells=density_matrix.n_s
    )

    assert csr.shape[:-1] == new_csr.shape
    assert np.allclose(csr.tocsr().toarray(), new_csr.toarray())


def test_nodes_and_edges_to_csr(
    density_matrix, density_config, density_data, density_z_table, symmetric
):
    csr = density_matrix._csr

    atom_labels_ptr = density_z_table.atom_block_pointer(density_data.point_types)
    edge_labels_ptr = density_z_table.edge_block_pointer(density_data.edge_types)
    edge_index = density_data.edge_index
    neigh_isc = density_data.neigh_isc
    if symmetric:
        edge_index = edge_index[:, ::2]
        neigh_isc = neigh_isc[::2]

    new_csr = nodes_and_edges_to_csr(
        density_data.point_labels,
        atom_labels_ptr,
        density_data.edge_labels,
        edge_index,
        edge_labels_ptr,
        edge_neigh_isc=neigh_isc,
        n_supercells=density_matrix.n_s,
        orbitals=density_config.atoms.orbitals,
        symmetrize_edges=symmetric,
    )

    assert csr.shape[:-1] == new_csr.shape
    assert np.allclose(csr.tocsr().toarray(), new_csr.toarray())


def test_nodes_and_edges_to_dm(
    density_matrix, density_config, density_data, density_z_table, symmetric
):
    atom_labels_ptr = density_z_table.atom_block_pointer(density_data.point_types)
    edge_labels_ptr = density_z_table.edge_block_pointer(density_data.edge_types)
    edge_index = density_data.edge_index
    neigh_isc = density_data.neigh_isc
    if symmetric:
        edge_index = edge_index[:, ::2]
        neigh_isc = neigh_isc[::2]

    new_csr = nodes_and_edges_to_csr(
        density_data.point_labels,
        atom_labels_ptr,
        density_data.edge_labels,
        edge_index,
        edge_labels_ptr,
        edge_neigh_isc=neigh_isc,
        n_supercells=density_matrix.n_s,
        orbitals=density_config.atoms.orbitals,
        symmetrize_edges=symmetric,
    )

    new_dm = csr_to_sisl_sparse_orbital(
        new_csr, geometry=density_matrix.geometry, sp_class=sisl.DensityMatrix
    )

    assert isinstance(new_dm, sisl.DensityMatrix)

    assert density_matrix.shape == new_dm.shape
    assert np.all(abs(new_dm - density_matrix)._csr.data < 1e-7)


def test_nodes_and_edges_to_dm_direct(
    density_matrix, density_data, density_z_table, symmetric
):
    atom_labels_ptr = density_z_table.atom_block_pointer(density_data.point_types)
    edge_labels_ptr = density_z_table.edge_block_pointer(density_data.edge_types)
    edge_index = density_data.edge_index
    neigh_isc = density_data.neigh_isc
    if symmetric:
        edge_index = edge_index[:, ::2]
        neigh_isc = neigh_isc[::2]

    new_dm = nodes_and_edges_to_sparse_orbital(
        density_data.point_labels,
        atom_labels_ptr,
        density_data.edge_labels,
        edge_index,
        edge_labels_ptr,
        edge_neigh_isc=neigh_isc,
        geometry=density_matrix.geometry,
        sp_class=sisl.DensityMatrix,
        symmetrize_edges=symmetric,
    )

    assert isinstance(new_dm, sisl.DensityMatrix)

    assert density_matrix.shape == new_dm.shape
    assert np.all(abs(new_dm - density_matrix)._csr.data < 1e-7)
