import pytest

from graph2mat import PointBasis, BasisTableWithEdges


@pytest.mark.parametrize(
    "basis_convention", ["cartesian", "spherical", "siesta_spherical"]
)
def test_init_table(basis_convention):
    point_1 = PointBasis("A", R=2, basis=[1], basis_convention=basis_convention)
    point_2 = PointBasis("B", R=5, basis=[2, 1], basis_convention=basis_convention)

    table = BasisTableWithEdges([point_1, point_2])

    assert table.basis_convention == basis_convention


def test_different_convention():
    point_1 = PointBasis("A", R=2, basis=[1], basis_convention="cartesian")
    point_2 = PointBasis("B", R=5, basis=[2, 1], basis_convention="spherical")

    with pytest.raises(AssertionError):
        table = BasisTableWithEdges([point_1, point_2])


def test_no_basis():
    point_1 = PointBasis("A", R=2, basis_convention="cartesian")
    point_2 = PointBasis("B", R=5)

    table = BasisTableWithEdges([point_1, point_2])
