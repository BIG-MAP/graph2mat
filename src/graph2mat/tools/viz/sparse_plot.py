import plotly.express as px
import plotly.graph_objects as go

import numpy as np
from typing import Dict, Union

import sisl
from scipy.sparse import issparse, spmatrix

from graph2mat import BasisConfiguration
from graph2mat.bindings.e3nn.irreps_tools import get_atom_irreps


def plot_basis_matrix(
    matrix: Union[np.ndarray, sisl.SparseCSR, spmatrix],
    configuration: Union[BasisConfiguration, sisl.Geometry, None] = None,
    point_lines: Union[bool, Dict] = False,
    basis_lines: Union[bool, Dict] = False,
    sc_lines: Union[bool, Dict] = False,
    colorscale: str = "RdBu",
    text: Union[bool, str] = False,
    basis_labels: bool = False,
) -> go.Figure:
    """Plots a matrix where rows and columns are spherical harmonics basis functions.

    Parameters
    -----------
    matrix:
        the matrix, either as a numpy array or as a sisl sparse matrix.
    configuration:
        Should contain the point coordinates and types associated with the matrix.
        Only needed if separator lines or basis labels are requested.
    point_lines:
        If a boolean, whether to draw lines separating points, using default styles.
        If a dict, draws the lines with the specified plotly line styles.
    basis_lines:
        If a boolean, whether to draw lines separating sets of basis functions, using default styles.
        If a dict, draws the lines with the specified plotly line styles.
    sc_lines:
        If a boolean, whether to draw lines separating the supercells, using default styles.
        If a dict, draws the lines with the specified plotly line styles.
    colorscale:
        A plotly colorscale.
    text:
        If a boolean, whether to show the value of each element as text on top of it, using plotly's
        default formatting.
        If a string, show text with the specified format. E.g. text=".3f" shows the value with three
        decimal places.
    basis_labels:
        Whether to label the axes with the basis function indices. If True, the labels will be of
        the form "P: (l, m)", where `P` is the index of the point and l and m are the indices of
        the spherical harmonic.
    """
    mode = "orbitals"

    if isinstance(matrix, sisl.SparseOrbital):
        if configuration is None:
            configuration = matrix.geometry

        matrix = matrix._csr
    elif isinstance(matrix, sisl.SparseAtom):
        if configuration is None:
            configuration = matrix.geometry

        matrix = matrix._csr
        mode = "atoms"

    geometry = configuration
    if isinstance(geometry, BasisConfiguration):
        geometry = geometry.to_sisl_geometry()

    if isinstance(matrix, sisl.SparseCSR):
        matrix = matrix.tocsr()

    if issparse(matrix):
        matrix = matrix.toarray()
        matrix[matrix == 0] = np.nan

    matrix = np.array(matrix)

    color_midpoint = None
    if np.sum(matrix < 0) > 0 and np.sum(matrix > 0) > 0:
        color_midpoint = 0

    fig = px.imshow(
        matrix,
        color_continuous_midpoint=color_midpoint,
        color_continuous_scale=colorscale,
        text_auto=text is True,
    )

    if point_lines is not False and mode == "orbitals":
        if point_lines is True:
            point_lines = {}

        point_lines = {"color": "orange", **point_lines}

        for atom_last_o in geometry.lasto[:-1]:
            line_pos = atom_last_o + 0.5
            fig.add_hline(
                y=line_pos,
                line=point_lines,
            )

            for i_s in range(geometry.n_s):
                fig.add_vline(x=line_pos + (i_s * geometry.no), line=point_lines)

    if basis_lines is not False and mode == "orbitals":
        if basis_lines is True:
            basis_lines = {}

        basis_lines = {"color": "black", "dash": "dot", **basis_lines}

        atom_irreps = [get_atom_irreps(atom) for atom in geometry.atoms.atom]

        curr_l = 0
        for atom_specie, atom_last_o in zip(geometry.atoms.specie, geometry.lasto):
            irreps = atom_irreps[atom_specie]

            for ir in irreps:
                m = ir[0]
                l = ir[1].l
                for _ in range(m):
                    curr_l += 2 * l + 1

                    if curr_l == atom_last_o + 1:
                        continue

                    line_pos = curr_l - 0.5

                    fig.add_hline(
                        y=line_pos,
                        line=basis_lines,
                    )

                    for i_s in range(geometry.n_s):
                        fig.add_vline(
                            x=line_pos + (i_s * geometry.no), line=basis_lines
                        )

    if sc_lines is not False:
        if sc_lines is True:
            sc_lines = {}

        sc_lines = {"color": "black", **sc_lines}
        sc_len = geometry.no if mode == "orbitals" else geometry.na

        for i_s in range(1, geometry.n_s):
            fig.add_vline(x=(i_s * sc_len) - 0.5, line=sc_lines, name=i_s)

    if isinstance(text, str):
        fig.update_traces(
            texttemplate="%{z:" + text + "}", selector={"type": "heatmap"}
        )

    if basis_labels:
        atoms_ticks = []
        atoms = geometry.atoms.atom
        for i, atom in enumerate(atoms):
            atom_ticks = []
            atoms_ticks.append(atom_ticks)
            for orb in atom.orbitals:
                atom_ticks.append(f"({orb.l}, {orb.m})")

        ticks = []
        for i, specie in enumerate(geometry.atoms.specie):
            ticks.extend([f"{i}: {orb}" for orb in atoms_ticks[specie]])

        fig.update_layout(
            yaxis_ticktext=ticks,
            yaxis_tickvals=np.arange(geometry.no),
            xaxis_ticktext=ticks,
            xaxis_tickvals=np.arange(geometry.no),
        )

    return fig
