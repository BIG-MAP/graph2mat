import copy

import numpy as np
import pytest
import torch
from e3nn import o3

from graph2mat import (
    BasisConfiguration,
    BasisTableWithEdges,
    MatrixDataProcessor,
    PointBasis,
    conversions,
)
from graph2mat.bindings.e3nn import E3nnGraph2Mat


@pytest.fixture(scope="module", params=["point_type", "basis_shape", "max"])
def basis_grouping(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def symmetric(request):
    return request.param


def get_rotation_matrix(data, table, alpha, beta, gamma):
    R_0 = 1
    R_1 = o3.wigner_D(1, alpha, beta, gamma)

    dim = table.basis_size[data.point_types].sum()
    big_R = np.zeros((dim, dim))

    for inv_els in (0, 1, 5, 9, 10):
        big_R[inv_els, inv_els] = R_0

    for start, end in ((2, 5), (6, 9), (11, 14)):
        big_R[start:end, start:end] = R_1

    return R_1, big_R


@pytest.fixture(scope="module", params=[False, True])
def with_no_basis(request):
    """Whether the test should include a point with no basis."""
    return request.param


def test_equivariance(basis_grouping, symmetric, with_no_basis):
    # The basis
    point_1 = PointBasis("A", R=3, basis="2x0e + 1o", basis_convention="cartesian")
    point_2 = PointBasis("B", R=3, basis="0e + 1o", basis_convention="cartesian")
    point_3 = PointBasis("C", R=3, basis="2x0e + 1o", basis_convention="cartesian")

    basis = [point_1, point_2, point_3]

    # Add an extra point with no basis
    if with_no_basis:
        point_4 = PointBasis("F", R=3, basis_convention="cartesian")
        basis.append(point_4)

    # The basis table.
    table = BasisTableWithEdges(basis)
    # The data processor.
    processor = MatrixDataProcessor(
        basis_table=table, symmetric_matrix=symmetric, sub_point_matrix=False
    )

    g2m = E3nnGraph2Mat(
        preprocessing_edges=None,
        unique_basis=table,
        irreps={"node_feats_irreps": o3.Irreps("0e + 1o")},
        symmetric=symmetric,
        basis_grouping=basis_grouping,
    )

    point_types = ["A", "B", "C"]
    positions = [[1, 1, 1], [2.0, 1, 1], [4.0, 1, 1]]
    # Add an extra point with no basis. To make sure that things work
    # in the most general case, we add it in the middle of the list.
    if with_no_basis:
        point_types.insert(1, "F")
        positions.insert(1, [6.0, 1, 1])

    config = BasisConfiguration(
        point_types=point_types,
        positions=np.array(positions),
        basis=basis,
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
    )

    conv = conversions.get_converter("basisconfiguration", "torch_basismatrixdata")
    data = conv(config, data_processor=processor)

    R1, big_R = get_rotation_matrix(
        data, table, torch.tensor(0), torch.tensor(0), torch.tensor(90 * np.pi / 180)
    )

    data_rot = copy.copy(data)

    data_rot.positions = data.positions @ R1.T

    def get_mat(data):
        node_feats = torch.concatenate(
            [data.point_types.reshape(-1, 1) + 1, data.positions], axis=1
        )
        nodes, edges = g2m(data, node_feats=node_feats)

        return processor.matrix_from_data(
            data,
            predictions={"node_labels": nodes, "edge_labels": edges},
            out_format="numpy",
        )

    pred = get_mat(data)
    pred_rot = get_mat(data_rot)

    post_rotated = big_R @ pred @ big_R.T

    max_diff = abs(post_rotated - pred_rot).max()

    assert max_diff < 1e-5, f"Equivariance error is too high: {max_diff:.2e}.\n"
