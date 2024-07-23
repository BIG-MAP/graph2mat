"""Utility tools to deal with e3nn irreps.

These are basically tools to convert from/to irreps.

They are currently not being used anywhere in `graph2mat`.
"""
from typing import Union, Sequence, Iterable

import sisl
import numpy as np

from e3nn import o3


def get_atom_irreps(atom: sisl.Atom):
    """For a given atom, returns the irreps representation of its basis.

    Parameters
    ----------
    atom: sisl.Atom
        The atom for which we want the irreps of its basis.

    Returns
    ----------
    o3.Irreps:
        the basis irreps.
    """

    if atom.no == 0:
        return o3.Irreps("")

    atom_irreps = []

    # Array that stores the number of orbitals for each l.
    # We allocate 8 ls, we will probably never need this much.
    n_ls = np.zeros(8)

    # Loop over all orbitals that this atom contains
    for orbital in atom.orbitals:
        # For each orbital, find its l quantum number
        # and increment the total number of orbitals for that l
        n_ls[orbital.l] += 1

    # We don't really want to know the number of orbitals for a given l,
    # but the number of SETS of orbitals. E.g. a set of l=1 has 3 orbitals.
    n_ls /= 2 * np.arange(8) + 1

    # Now just loop over all ls, and intialize as much irreps as we need
    # for each of them. We build a list of tuples (n_irreps, (l, parity))
    # to pass it to o3.Irreps.
    for l, n_l in enumerate(n_ls):
        if n_l != 0:
            atom_irreps.append((int(n_l), (l, (-1) ** l)))

    return o3.Irreps(atom_irreps)


def get_atom_from_irreps(
    irreps: Union[o3.Irreps, str],
    orb_kwargs: Union[Iterable[dict], dict] = {},
    atom_args: Sequence = (),
    **kwargs,
):
    """Returns a sisl atom with the basis specified by irreps."""
    if isinstance(orb_kwargs, dict):
        orb_kwargs = [orb_kwargs] * len(o3.Irreps(irreps).ls)

    orbitals = []
    for orbital_l, orbital_kwargs in zip(o3.Irreps(irreps).ls, orb_kwargs):
        if len(orbital_kwargs) == 0:
            orbital_kwargs = {
                "rf_or_func": None,
            }

        for m in range(-orbital_l, orbital_l + 1):
            orbital = sisl.SphericalOrbital(l=orbital_l, m=m, **orbital_kwargs)

            orbitals.append(orbital)

    if len(atom_args) == 0:
        kwargs = {
            "Z": 1,
            **kwargs,
        }

    return sisl.Atom(*atom_args, orbitals=orbitals, **kwargs)
