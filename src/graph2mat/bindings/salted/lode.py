from typing import Literal, Optional, List

import numpy as np

from salted import sph_utils
from salted.lib import equicomb

from sympy.physics.wigner import wigner_3j

__all__ = ["E3nnLODE"]

###############################
#      HELPER FUNCTIONS
###############################


def get_wigner3j(llmax, llvec, lam):
    """Compute and save Wigner-3J symbols needed for symmetry-adapted combination

    Copied from salted/wigner.py
    """

    wigners = []

    for il in range(llmax):
        l1 = int(llvec[il, 0])
        l2 = int(llvec[il, 1])
        for imu in range(2 * lam + 1):
            mu = imu - lam
            for im1 in range(2 * l1 + 1):
                m1 = im1 - l1
                m2 = m1 - mu
                if abs(m2) <= l2:
                    im2 = m2 + l2
                    w3j = wigner_3j(lam, l2, l1, mu, m2, -m1) * (-1.0) ** (m1)
                    wigners.append(float(w3j))

    return np.array(wigners)


###############################
#   FUNCTION TO COMPUTE LODE
###############################


def LODE(
    structure: "ase.Atoms",
    lmax: int,
    params_rep1: dict = {},
    params_rep2: dict = {},
    rep1: Literal["rho", "V"] = "rho",
    rep2: Literal["rho", "V"] = "V",
    neighspe1: Optional[List[str]] = None,
    neighspe2: Optional[List[str]] = None,
):
    """Computes the lode descriptors for one structure.

    Parameters
    ----------
    structure:
        ASE atoms object containing the structure for which to compute LODE.
    lmax:
        Maximum order of spherical harmonics of the LODE descriptor.
    params_rep1:
        Hyperparameters for the first representation.
    params_rep2:
        Hyperparameters for the second representation.
    rep1:
        Type of the first representation.
    rep1:
        Type of the second representation.
    neighspe1:
        List of species for the first descriptor.
        If None, the full list of species is taken.
    neighspe2:
        List of species for the second descriptor.
        If None, the full list of species is taken.
    """
    # Gather some information about the structure
    natoms = len(structure)
    species = list(set(structure.get_chemical_symbols()))

    # If the species for the representation is not set, set it to all
    # species
    if neighspe1 is None:
        neighspe1 = species
    if neighspe2 is None:
        neighspe2 = species

    # Get number of species for each descriptor
    nspe1, nspe2 = len(neighspe1), len(neighspe2)

    # Extra hyper parameters for computing rho and V descriptors
    rho_params = {
        "center_atom_weight": 1.0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-6}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
    }
    V_params = {
        "potential_exponent": 1,
        "center_atom_weight": 1.0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-6}},
    }

    # I don't know why this is used, looking at get_representation_coeffs code
    # it doesn't look like it is being used
    rank = 1

    nang1, nrad1 = params_rep1["max_angular"], params_rep1["max_radial"]
    nang2, nrad2 = params_rep2["max_angular"], params_rep2["max_radial"]

    # Compute the two representations.
    omega1 = sph_utils.get_representation_coeffs(
        structure,
        rep1,
        HYPER_PARAMETERS_DENSITY={**rho_params, **params_rep1},
        HYPER_PARAMETERS_POTENTIAL={**V_params, **params_rep1},
        rank=rank,
        neighspe=neighspe1,
        species=species,
        nang=nang1,
        nrad=nrad1,
        natoms=natoms,
    )
    omega2 = sph_utils.get_representation_coeffs(
        structure,
        rep2,
        HYPER_PARAMETERS_DENSITY={**rho_params, **params_rep2},
        HYPER_PARAMETERS_POTENTIAL={**V_params, **params_rep2},
        rank=rank,
        neighspe=neighspe2,
        species=species,
        nang=nang2,
        nrad=nrad2,
        natoms=natoms,
    )

    # Reshape arrays of expansion coefficients for optimal Fortran indexing
    v1 = np.transpose(omega1, (2, 0, 3, 1))
    v2 = np.transpose(omega2, (2, 0, 3, 1))

    # Compute equivariant features for the given structure
    power = {}
    for lam in range(lmax + 1):
        [llmax, llvec] = sph_utils.get_angular_indexes_symmetric(lam, nang1, nang2)

        # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
        wigner3j = get_wigner3j(llmax, llvec, lam)
        wigdim = wigner3j.size

        # Compute complex to real transformation matrix for the given lambda value
        c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]

        # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
        featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
        p = equicomb.equicomb(
            natoms,
            nang1,
            nang2,
            nspe1 * nrad1,
            nspe2 * nrad2,
            v1,
            v2,
            wigdim,
            wigner3j,
            llmax,
            llvec.T,
            lam,
            c2r,
            featsize,
        )
        p = np.transpose(p, (2, 0, 1))

        # Fill vector of equivariant descriptor
        if lam == 0:
            power[lam] = p.reshape(natoms, featsize)
        else:
            power[lam] = p.reshape(natoms, 2 * lam + 1, featsize)

    return power


from e3nn import o3
import torch
import ase
from graph2mat.core.data.basis import get_change_of_basis


class E3nnLODE(torch.nn.Module):
    def __init__(
        self,
        basis,
        lmax: int,
        params_rep1: dict = {},
        params_rep2: dict = {},
        rep1: Literal["rho", "V"] = "rho",
        rep2: Literal["rho", "V"] = "V",
        neighspe1: Optional[List[str]] = None,
        neighspe2: Optional[List[str]] = None,
    ):
        super().__init__()

        self.change_of_basis = get_change_of_basis("cartesian", "spherical")[1]

        # Sanitize species as ASE chemical symbols
        self.neighspe = np.array(
            ase.Atoms(
                positions=[[0, 0, 0]] * len(basis),
                symbols=[point.type for point in basis],
            ).get_chemical_symbols()
        )

        self.lmax = lmax
        self.rep1 = rep1
        self.rep2 = rep2
        self.neighspe1 = neighspe1 if neighspe1 is not None else self.neighspe
        self.neighspe2 = neighspe2 if neighspe2 is not None else self.neighspe
        self.params_rep1 = params_rep1
        self.params_rep2 = params_rep2

        nspe1 = len(self.neighspe1)
        nspe2 = len(self.neighspe2)

        all_irreps = []
        for l in range(self.lmax + 1):
            [llmax, llvec] = sph_utils.get_angular_indexes_symmetric(
                l, params_rep1["max_angular"], params_rep2["max_angular"]
            )
            mul = (
                nspe1
                * nspe2
                * params_rep1["max_radial"]
                * params_rep2["max_radial"]
                * llmax
            )

            all_irreps.append((mul, (l, (-1) ** l)))

        self.irreps_out = o3.Irreps(all_irreps)

    def forward(self, data):
        examples = data[:]

        all_lodes = {l: [] for l in range(self.lmax + 1)}

        for example in examples:
            ats = ase.Atoms(
                positions=example.positions @ self.change_of_basis.T,
                symbols=self.neighspe[example.point_types],
                cell=example.cell @ self.change_of_basis.T,
                pbc=True,
            )

            lode = LODE(
                ats,
                self.lmax,
                self.params_rep1,
                self.params_rep2,
                self.rep1,
                self.rep2,
                self.neighspe1,
                self.neighspe2,
            )

            for k in lode:
                all_lodes[k].append(lode[k])

        def _san_array(array):
            array = np.concatenate(array)

            if array.ndim == 3:
                array = array.transpose((0, 2, 1))

            return array.reshape(array.shape[0], -1)

        all_lodes = {k: _san_array(v) for k, v in all_lodes.items()}

        node_feats = np.concatenate(list(all_lodes.values()), axis=1)

        return torch.tensor(node_feats, dtype=torch.get_default_dtype())
