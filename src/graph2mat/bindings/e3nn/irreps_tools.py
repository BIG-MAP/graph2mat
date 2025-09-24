"""Utility tools to deal with e3nn irreps.

These are basically tools to convert from/to irreps.

They are currently not being used anywhere in `graph2mat`.
"""
from typing import Iterable, Sequence, Union

import numpy as np
import sisl
import torch
from e3nn import o3

from graph2mat.bindings.e3nn._stored_rtps import ALL_RTPS


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


def expand_irreps(irreps: Union[o3.Irreps, str]) -> o3.Irreps:
    """Expands an irreps representation to have all irreps with multiplicity 1.

    For example, "2x0e + 1x1o" becomes "0e + 0e + 1o".

    Parameters
    ----------
    irreps:
        The irreps to expand.
    """
    irreps = o3.Irreps(irreps)
    expanded_irreps = o3.Irreps()

    for irrep in irreps:
        expanded_irreps += sum([irrep.ir] * irrep.mul, o3.Irreps())

    return expanded_irreps


class ReducedTensorProducts:
    """Class to initialize reduced tensor products much faster than e3nn.

    Here we take advantage of the fact that we only need to support a few
    cases, and we use possibly precomputed reduced tensor products to accelerate
    the initialization dramatically.

    The interface is exactly the same as e3nn.o3.ReducedTensorProducts.
    """

    def __new__(cls, formula, **kwargs):
        if formula not in ("ij", "ij=ji") or any(k not in kwargs for k in "ij"):
            return o3.ReducedTensorProducts(formula, **kwargs)

        return super().__new__(cls)

    def __init__(self, formula, i, j):
        # Expand the irreps so that we have all irreps with multiplicity 1
        i_irreps = expand_irreps(i)
        j_irreps = expand_irreps(j)

        # Check that the formula is compatible with the irreps
        if formula == "ij=ji" and i != j:
            raise ValueError(
                f"Formula ij=ji requires irreps of i ({i}) == irreps of j ({j})"
            )

        # Init the variables where we will accumulate all the RTPs
        all_change_of_basis = []
        irreps_out = o3.Irreps("")

        # Loop through rows and columns of the tensor product
        row = 0
        for i, i_irrep in enumerate(i_irreps):
            col = 0
            for j, j_irrep in enumerate(j_irreps):
                # If ij=ji, we only compute the upper triangular part
                if formula == "ij=ji" and i > j:
                    col += j_irrep.dim
                    continue

                # Get the RTP for this pair of irreps
                if formula == "ij=ji" and i == j:
                    rtp = ALL_RTPS[str(i_irrep.ir), str(j_irrep.ir), "ij=ji"]
                else:
                    rtp = ALL_RTPS[str(i_irrep.ir), str(j_irrep.ir), "ij"]

                # Loop through all the output irreps and store the corresponding change of basis
                # We need to separate the change of basis for each output irrep because we will need
                # to re-order them at the end.
                ir_start = 0
                for out_ir in rtp["irreps_out"]:
                    ir_end = ir_start + out_ir.dim

                    all_change_of_basis.append(
                        (
                            (row, row + i_irrep.dim, col, col + j_irrep.dim),
                            rtp["change_of_basis"][ir_start:ir_end],
                        )
                    )

                    ir_start = ir_end

                # Accumulate the output irreps
                irreps_out += rtp["irreps_out"]

                col += j_irrep.dim

            row += i_irrep.dim

        # Sort and simplify the output irreps
        irreps_sort = irreps_out.sort()
        self.irreps_out = irreps_sort.irreps.simplify()

        # And then sort accordingly the change of basis, and put everything in a single tensor
        change_of_basis = torch.zeros(
            irreps_out.dim, i_irreps.dim, j_irreps.dim, dtype=torch.get_default_dtype()
        )
        start_index = 0
        for i_irrep in irreps_sort.inv:
            (row, end_row, col, end_col), ir_change_of_basis = all_change_of_basis[
                i_irrep
            ]

            end_index = start_index + ir_change_of_basis.shape[0]

            change_of_basis[
                start_index:end_index, row:end_row, col:end_col
            ] = ir_change_of_basis

            start_index = end_index

        # Symmetrize (and normalize) the change of basis in case that the tensor product is symmetric
        if formula == "ij=ji":
            change_of_basis = change_of_basis + change_of_basis.transpose(1, 2)
            change_of_basis = change_of_basis / torch.linalg.norm(
                change_of_basis, dim=(1, 2)
            ).reshape(-1, 1, 1)

        self.change_of_basis = change_of_basis
