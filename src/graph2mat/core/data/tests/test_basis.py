import pytest

import numpy as np
import itertools

import sisl

from graph2mat import PointBasis
from graph2mat.core.data.basis import get_atom_basis, get_change_of_basis


def test_simplest():
    basis = PointBasis("A", basis_convention="spherical", basis="3x0e + 2x1o", R=5)


def test_list_basis():
    basis_str = PointBasis("A", basis_convention="spherical", basis="3x0e + 2x1o", R=5)

    basis_list = PointBasis("A", basis_convention="spherical", basis=[3, 2], R=5)

    assert basis_str == basis_list


def test_siesta_convention():
    basis = PointBasis(
        "A", basis_convention="siesta_spherical", basis="3x0e + 2x1o", R=5
    )


def test_no_basis():
    basis = PointBasis("A", R=5)


def test_multiple_R():
    basis = PointBasis(
        "A",
        basis_convention="spherical",
        basis="3x0e + 2x1o",
        R=np.array([5, 5, 5, 3, 3, 3, 3, 3, 3]),
    )

    # Wrong number of Rs
    with pytest.raises(AssertionError):
        basis = PointBasis(
            "A",
            basis_convention="spherical",
            basis="3x0e + 2x1o",
            R=np.array([5, 5, 5, 3, 3]),
        )


def test_from_sisl_atom():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("2p{ax}", R=4) for ax in "xyz"])

    basis = PointBasis.from_sisl_atom(atom)

    assert basis.type == 1
    assert basis.basis_convention == "siesta_spherical"
    # assert basis.irreps == o3.Irreps("1x1o")
    assert isinstance(basis.R, np.ndarray)
    assert np.all(basis.R == 4)


def test_to_sisl_atom():
    basis = PointBasis(
        "A",
        basis_convention="siesta_spherical",
        basis="3x0e + 2x1o",
        R=np.array([5, 5, 5, 3, 3, 3, 3, 3, 3]),
    )

    atom = basis.to_sisl_atom()

    isinstance(atom, sisl.Atom)
    assert len(atom.orbitals) == 9

    for orbital in atom.orbitals[:3]:
        assert orbital.l == 0
        assert orbital.R == 5

    for orbital in atom.orbitals[3:]:
        assert orbital.l == 1
        assert orbital.R == 3


def test_get_atom_basis():
    atom = sisl.Atom(
        1,
        orbitals=[
            sisl.AtomicOrbital("2s", R=4),
            *[sisl.AtomicOrbital(f"2p{ax}", R=4) for ax in "xyz"],
            *[sisl.AtomicOrbital(f"2p{ax}Z2", R=4) for ax in "xyz"],
        ],
    )

    atom_basis = get_atom_basis(atom)

    assert atom_basis == [(1, 0, 1), (2, 1, -1)]


def test_change_of_basis_consistent():
    conventions = ["spherical", "siesta_spherical", "cartesian"]
    for original, target in itertools.combinations_with_replacement(conventions, 2):
        a, b = get_change_of_basis(original, target)
        c, d = get_change_of_basis(target, original)

        assert np.allclose(a, d), f"{original}-{target}"
        assert np.allclose(b, c), f"{original}-{target}"


def test_change_of_basis_works():
    """Check that we can go through different basis, go back
    to the original basis and get the original array."""

    array = np.random.rand(4, 3)

    cob, _ = get_change_of_basis("cartesian", "siesta_spherical")

    in_siesta = array @ cob.T

    cob, _ = get_change_of_basis("siesta_spherical", "spherical")

    spherical = in_siesta @ cob.T

    cob, _ = get_change_of_basis("spherical", "cartesian")

    array_again = spherical @ cob.T

    assert np.allclose(array, array_again)
