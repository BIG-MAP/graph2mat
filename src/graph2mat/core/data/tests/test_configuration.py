import pytest
import numpy as np

from graph2mat import BasisConfiguration, OrbitalConfiguration, PointBasis


@pytest.mark.parametrize("config_cls", [BasisConfiguration, OrbitalConfiguration])
def test_init_configuration(config_cls):
    # The basis
    point_1 = PointBasis("A", R=2, basis=[1], basis_convention="spherical")
    point_2 = PointBasis("B", R=5, basis=[2, 1], basis_convention="spherical")

    positions = np.array([[0, 0, 0], [6.0, 0, 0], [9, 0, 0]])

    # Initialize configuration
    config_cls(
        point_types=["A", "B", "A"],
        positions=positions,
        basis=[point_1, point_2],
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
    )
