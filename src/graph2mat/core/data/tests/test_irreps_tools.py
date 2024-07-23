import sisl

from e3nn.o3 import Irreps

from graph2mat.data.irreps_tools import get_atom_irreps, get_atom_from_irreps


def test_get_atom_irreps():
    atom = sisl.Atom(
        1,
        orbitals=[
            sisl.AtomicOrbital("2s", R=4),
            *[sisl.AtomicOrbital(f"2p{ax}", R=4) for ax in "xyz"],
            *[sisl.AtomicOrbital(f"2p{ax}Z2", R=4) for ax in "xyz"],
        ],
    )

    atom_irreps = get_atom_irreps(atom)

    assert atom_irreps == Irreps("1x0e+2x1o")


def test_get_atom_from_irreps():
    atom_irreps = Irreps("1x0e+2x1o")

    atom = get_atom_from_irreps(atom_irreps)

    assert isinstance(atom, sisl.Atom)
    assert len(atom.orbitals) == 7

    assert atom.orbitals[0].l == 0

    for orbital in atom.orbitals[1:]:
        assert orbital.l == 1
