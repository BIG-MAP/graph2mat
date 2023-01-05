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
    n_ls /= (2*np.arange(8) + 1)

    # Now just loop over all ls, and intialize as much irreps as we need
    # for each of them. We build a list of tuples (n_irreps, (l, parity))
    # to pass it to o3.Irreps.
    for l, n_l in enumerate(n_ls):
        if n_l != 0:
            atom_irreps.append((int(n_l), (l, (-1)**l)))

    return o3.Irreps(atom_irreps)