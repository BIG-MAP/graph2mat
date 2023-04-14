"""Implements a flask server that accepts requests to predict matrices."""

from pathlib import Path
from typing import Callable, Dict, Literal

import sisl
import numpy as np

from e3nn_matrix.data.configuration import load_orbital_config_from_run
from e3nn_matrix.torch.data import OrbitalMatrixData
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges
#from e3nn_matrix.bindings.mace.lit import LitOrbitalMatrixMACE

from flask import Flask, request

def predictions_server_app(
    prediction_function: Callable[[OrbitalMatrixData], Dict[str, np.ndarray] ] ,
    z_table: AtomicTableWithEdges,
    sub_atomic_matrix: bool,
    symmetric_matrix: bool,
    out_matrix: Literal["density_matrix", "energy_density_matrix", "hamiltonian"],
    output_file_name: str = "predicted.DM",
) -> Flask:
    """Creates a flask app to listen to requests and predict matrices.

    Parameters
    ----------
    prediction_function: Callable[[OrbitalMatrixData], Dict[str, np.ndarray]]
        Function that predicts the orbital matrix, given an empty OrbitalMatrixData object
        that contains information about the structure only.
    z_table: AtomicTableWithEdges
        Atomic table that contains information about the atomic species, with basis orbitals,
        that the predicting model knows about.
    sub_atomic_matrix: bool
        Whether the model used has learnt to predict the difference between the atomic
        and the final matrix (True) or just the total matrix (False).
    symmetric_matrix: bool
        Whether the model is trained on symmetric matrices.
    out_matrix: Literal["density_matrix", "energy_density_matrix", "hamiltonian"]
        The type of matrix that the model predicts. Important to get the appropiate sisl class that
        writes the specific matrix type to files.
    output_file_name: str, optional
        Name of the output file where the predictions will be written by default, i.e. if no 
        path is specified in the request.
    """

    app = Flask(__name__)

    # Pick the class for the output matrix in sisl.
    matrix_cls = {
        "density_matrix": sisl.DensityMatrix,
        "energy_density_matrix": sisl.EnergyDensityMatrix,
        "hamiltonian": sisl.Hamiltonian,
    }[out_matrix]

    @app.route('/write_prediction', methods=['GET'])
    def write_prediction():
        """Given a geometry file, writes the predicted matrix to a file.

        The request needs to contain the following arguments:
        geometry: str or Path
            Path to the file that contains the geometry that we want to predict. Any
            file format that sisl can read is supported.
        output: str or Path, optional
            Path to the file that we want to write the predicted matrix to. If not
            specified, the default name for the app will be used.

            NOTE: If the output is not an absolute path, it will be interpreted as
            relative to the directory of the input file.

            Also, if the output file already exists, an error will be raised unless
            allow_overwrite is set to True.
        allow_overwrite: bool, optional
            If True, allow overwriting of existing files. Default is False.

        Returns
        -------
        str
            Absolute path to the file where the matrix was written.
        """
        # Get all the arguments of the requests and start parsing them.
        args = request.args

        # First, the input file, which we resolve to an absolute path to avoid
        # ambiguity.
        input_file = args.get('geometry', 'siesta.XV')
        runfile = Path(input_file).resolve()

        # Then the output path where we should write the matrix.
        out_file = args.get('output', output_file_name)
        out_file = Path(out_file).resolve()
        if not out_file.is_absolute():
            out_file = runfile.parent / out_file
        out_file = Path(out_file).resolve()

        # Also, whether overwriting an existing file is allowed.
        allow_overwrite = args.get('allow_overwrite', False)
        
        # Load the configuration object, containing information about the structure.
        config = load_orbital_config_from_run(runfile, out_matrix=None)

        # Initialize an empty matrix data object.
        matrix_data = OrbitalMatrixData.from_config(
            config,
            z_table=z_table,
            sub_atomic_matrix=sub_atomic_matrix,
            symmetric_matrix=symmetric_matrix,
        )

        # Ask for a prediction on this structure.
        prediction = prediction_function(matrix_data)
        
        if prediction is not None:
            matrix_data.atom_labels = prediction['node_labels']
            matrix_data.edge_labels = prediction['edge_labels']
        
        sparse_orbital_matrix = matrix_data.to_sparse_orbital_matrix(z_table, matrix_cls, symmetric_matrix, sub_atomic_matrix)

        if allow_overwrite and out_file.exists():
            raise ValueError(f"Output file {out_file} already exists and overwrite is not allowed.")
        
        # And write the matrix to it.
        sparse_orbital_matrix.write(out_file)

        return str(out_file)

    return app