import pytest

import numpy as np

from graph2mat import (
    PointBasis,
    BasisTableWithEdges,
    MatrixDataProcessor,
    BasisConfiguration,
    BasisMatrixData,
    OrbitalConfiguration,
)


@pytest.fixture(scope="module")
def positions():
    return np.array([[0, 0, 0], [6.0, 0, 0]])


@pytest.fixture(scope="module", params=["cartesian", "spherical", "siesta_spherical"])
def basis_convention(request):
    return request.param


@pytest.fixture(scope="module")
def basis_table(basis_convention):
    point_1 = PointBasis("A", R=2, basis=[1], basis_convention=basis_convention)
    point_2 = PointBasis("B", R=5, basis=[2, 1], basis_convention=basis_convention)

    return BasisTableWithEdges([point_1, point_2])


@pytest.mark.parametrize("config_cls", [BasisConfiguration, OrbitalConfiguration])
@pytest.mark.parametrize("new_method", ["from_config", "new"])
def test_init_data(positions, basis_table, basis_convention, new_method, config_cls):
    # The data processor.
    processor = MatrixDataProcessor(
        basis_table=basis_table, symmetric_matrix=True, sub_point_matrix=False
    )

    config = config_cls(
        point_types=["A", "B"],
        positions=positions,
        basis=basis_table,
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
    )

    # Test from_config method
    new = getattr(BasisMatrixData, new_method)
    data = new(config, processor)

    if basis_convention == "cartesian":
        assert np.all(data.positions == positions)
    else:
        assert (data.positions != positions).sum() == 2
