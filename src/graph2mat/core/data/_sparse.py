"""Functions whose performance is critical in sparse conversions.

This file can either be used directly by the python interpreter
or cythonized for increased performance.

The cython compilation assumes that cython.int is int32 so that
we don't have to cimport numpy. This might not be true in some
machines (?).
"""

import numpy as np

import cython


def _csr_to_block_dict(
    data: cython.numeric[:],
    ptr: cython.int[:],
    cols: cython.int[:],
    atom_first_orb: cython.int[:],
    orbitals: cython.int[:],
    n_atoms: cython.int,
):
    # --- Cython annotations for increased performance (ignored if not compiled with cython)
    atom_i: cython.int
    atomi_firsto: cython.int
    atomi_lasto: cython.int
    atomi_norbs: cython.int

    orbital_i: cython.int
    atom_j: cython.int
    orbital_j: cython.int

    no: cython.int

    ival: cython.int
    val: cython.numeric
    sc_col: cython.int
    col: cython.int
    i_sc: cython.int
    row: cython.int

    rc_to_atom_index: cython.int[:]
    rc_to_orbital_index: cython.int[:]
    # ------ End of cython annotations.

    # Mapping from row/column index to atom index
    rc_to_atom_index = np.concatenate(
        [np.ones(o, dtype=np.int32) * i for i, o in enumerate(orbitals)]
    )
    # Mapping from row/column index to orbital index within atom
    rc_to_orbital_index = np.concatenate(
        [np.arange(o) for o in orbitals], dtype=np.int32
    )

    no = atom_first_orb[n_atoms]

    block_dict = {}
    for atom_i in range(n_atoms):
        # Get the orbital limits of the block
        atomi_firsto = atom_first_orb[atom_i]
        atomi_lasto = atom_first_orb[atom_i + 1]
        # And the size of the block
        atomi_norbs = atomi_lasto - atomi_firsto

        # Loop over rows in this atom.
        for orbital_i in range(atomi_norbs):
            # Get the row index
            row = atomi_firsto + orbital_i

            for ival in range(ptr[row], ptr[row + 1]):
                val = data[ival]
                sc_col = cols[ival]

                # Sisl SparseCSR allocates space in advance for values. Values that are
                # allocated but have not been set contain a col of -1.
                if sc_col < 0:
                    break

                col = sc_col % no
                i_sc = sc_col // no

                atom_j = rc_to_atom_index[col]
                orbital_j = rc_to_orbital_index[col]
                try:
                    block_dict[atom_i, atom_j, i_sc][orbital_i, orbital_j] = val
                except KeyError:
                    block_dict[atom_i, atom_j, i_sc] = np.full(
                        (orbitals[atom_i], orbitals[atom_j]), np.nan
                    )
                    block_dict[atom_i, atom_j, i_sc][orbital_i, orbital_j] = val

    return block_dict
