import numpy as np
import sisl
import torch

from scipy.spatial.transform import Rotation

from e3nn_matrix.data.configuration import OrbitalConfiguration
from e3nn_matrix.torch.data import OrbitalMatrixData
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges
from e3nn_matrix.data.irreps_tools import get_atom_irreps
from e3nn_matrix.data.sparse import nodes_and_edges_to_coo

from e3nn_matrix.bindings.mace.lit import LitOrbitalMatrixMACE

import pytest


s_orbitals = [sisl.AtomicOrbital(n=1, l=0, R=2.5)]
p_orbitals = [sisl.AtomicOrbital(n=2, l=1, m=-1, R=3), sisl.AtomicOrbital(n=2, l=1, m=0, R=3), sisl.AtomicOrbital(n=2, l=1, m=1, R=3)]
d_orbitals = [
    sisl.AtomicOrbital(n=2, l=2, m=-2, R=3), sisl.AtomicOrbital(n=2, l=2, m=-1, R=3), sisl.AtomicOrbital(n=2, l=2, m=0, R=3),
    sisl.AtomicOrbital(n=2, l=2, m=1, R=3), sisl.AtomicOrbital(n=2, l=2, m=2, R=3),
]

@pytest.fixture(scope="module", params=("same_basis", "DZP"))
def basis_shape(request):
    return request.param

@pytest.fixture(scope="module")
def z_table(basis_shape):
    if basis_shape == "same_basis":
        H = sisl.Atom("H", orbitals=[*s_orbitals, *p_orbitals])
        O = sisl.Atom("O", orbitals=[*s_orbitals, *p_orbitals])
    elif basis_shape == "DZP":
        H = sisl.Atom("H", orbitals=[*s_orbitals, *s_orbitals, *p_orbitals, *p_orbitals])
        O = sisl.Atom("O", orbitals=[*s_orbitals, *s_orbitals, *p_orbitals, *p_orbitals, *d_orbitals])
    else:
        raise ValueError("basis_shape was provided a wrong value")

    return AtomicTableWithEdges([H, O])

@pytest.fixture(scope="module")
def model(z_table):
    return LitOrbitalMatrixMACE(
        z_table=z_table,
        symmetric_matrix=True,
        avg_num_neighbors=1,
        correlation=2,
        max_ell=2,
        num_interactions=2,
        hidden_irreps="10x0e + 10x1o + 10x2e",
        edge_hidden_irreps="4x0e + 4x1o + 4x2e",
    )

@pytest.fixture(scope="module", params=("bimolec", "square"))
def geometry(request, z_table):
    [H, O] = z_table.atoms
    system = request.param
    if system == "bimolec":
        geom = sisl.Geometry([[0, 0, 0], [1, 0, 0]], atoms=[H, O], sc=[20, 20, 20])
    elif system == "square":
        geom = sisl.Geometry([[0,0,0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], atoms=[H, O, H, O], sc=[20,20,20])
    else:
        raise ValueError("system was provided a wrong value")

    return geom

def test_model_equivariance(model, z_table, geometry):

    geom = geometry

    # Helper function that returns the predicted matrix from a geometry.
    def get_matrix(geom):
        config = OrbitalConfiguration.from_geometry(geom)
        data = OrbitalMatrixData.from_config(config, z_table, symmetric_matrix=True)

        out = model.model(data)

        return nodes_and_edges_to_coo(
            out['node_labels'].detach().numpy(), z_table.atom_block_pointer(data.atom_types),
            out['edge_labels'].detach().numpy(), data.edge_index[:, ::2].numpy(), z_table.edge_block_pointer(data.edge_types[::2]),
            geom.orbitals, symmetrize_edges=True
        ).toarray()

    atom_irreps = [get_atom_irreps(atom) for atom in z_table.atoms]
    irreps = None
    for at in geom.atoms.specie:
        if irreps is None:
           irreps = atom_irreps[at]
        else:
           irreps = irreps + atom_irreps[at]
    # And the rotated one
    R = Rotation.from_euler("xyz", [20, 30, 50]).as_matrix()
    rot_geom = geom.copy()
    rot_geom.xyz = rot_geom.xyz @ R.T

    # Get the predicted matrices for both
    out = get_matrix(geom)
    rot_out = get_matrix(rot_geom)

    # Get the matrix that rotates the unrotated geometry output to get the expected output.
    # Note that a change of basis to the spherical harmonics is needed.
    basis_change = OrbitalMatrixData._change_of_basis
    D = irreps.D_from_matrix(basis_change @ torch.tensor(R, dtype=torch.get_default_dtype()) @ basis_change.T)

    # Check that the expected rotated output is the same as the produced output.
    # We set the tolerance to 50 pico, which is quite high, but it is the precision that we have seen to acheive with
    # float32. With float64 you can go higher.
    assert np.allclose(D @ out @ D.T, rot_out, atol=5e-5)