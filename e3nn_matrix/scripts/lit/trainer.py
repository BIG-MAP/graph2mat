from pathlib import Path
from typing import Optional, Union
import socket
import logging

from pytorch_lightning import Trainer, LightningModule, LightningDataModule
import torch
import sisl

from e3nn_matrix.data.configuration import load_orbital_config_from_run
from e3nn_matrix.torch.data import OrbitalMatrixData
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges
#from e3nn_matrix.bindings.mace.lit import LitOrbitalMatrixMACE

def _write_matrix_data_to_file(filename: Union[Path,str], matrix_data: OrbitalMatrixData, z_table: AtomicTableWithEdges, matrix_cls, symmetric_matrix: bool, sub_atomic_matrix: bool, prediction=None):
    if prediction is not None:
        matrix_data.atom_labels = prediction['node_labels']
        matrix_data.edge_labels = prediction['edge_labels']

    sparse_orbital_matrix = matrix_data.to_sparse_orbital_matrix(z_table, matrix_cls, symmetric_matrix, sub_atomic_matrix)
    # And write the matrix to it.
    sparse_orbital_matrix.write(filename)


class MatrixTrainer(Trainer):
    def serve(
        self,
        model: LightningModule,
        ckpt_path: str,
        output_file: str = "predicted.DM",
        datamodule: Optional[LightningDataModule] = None,
        host="localhost",
        port: int=56000,
        allow_overwrite: bool = False,
    ):
        r"""
        Runs a server that provides predictions with a trained model.

        The implementation does not follow the regular lightning loop,
        so callbacks etc. will not work.

        Args:
            model: Model definition
            ckpt_path: Path to the checkpoint file
            datamodule: Not used, but cli will provide this argument
            listen_port: Which port to use for connections
        """
        from flask import Flask, request

        # Instantiate new model from checkpoint. The hyperparameters of the
        # current model are not from the checkpoint, except for the z_table,
        # so we need to instantiate a new model by calling the
        # load_from_checkpoint class method
        model = model.load_from_checkpoint(ckpt_path, z_table=model.z_table)
        model.eval()

        # Find out which matrix class we should use based on what matrix type the data
        # has been trained on.
        if datamodule is not None:
            datamodule = datamodule.load_from_checkpoint(ckpt_path)
            out_matrix = datamodule.out_matrix
        else:
            raise RuntimeError("Failed to determine out_matrix type")
        assert out_matrix is not None
        matrix_cls = {
            "density_matrix": sisl.DensityMatrix,
            "energy_density_matrix": sisl.EnergyDensityMatrix,
            "hamiltonian": sisl.Hamiltonian,
            "dynamical_matrix": sisl.DynamicalMatrix,
        }[out_matrix]

        app = Flask(__name__)

        @app.route('/predict', methods=['GET'])
        def search():
            args = request.args
            input_file = args.get('geometry', 'siesta.XV')
            out_file = args.get('output', output_file)

            runfile = Path(input_file)
            config = load_orbital_config_from_run(runfile, out_matrix=None)
            matrix_data = OrbitalMatrixData.from_config(
                config,
                z_table=model.z_table,
                sub_atomic_matrix=datamodule.sub_atomic_matrix,
                symmetric_matrix=model.hparams.symmetric_matrix,
            )
            with torch.no_grad():
                prediction = model(matrix_data)

            path = matrix_data.metadata["path"]
            output_path = path.parent / out_file

            if allow_overwrite and output_path.exists():
                raise ValueError(f"Output file {output_path} already exists")

            _write_matrix_data_to_file(
                output_path,
                matrix_data,
                model.z_table,
                matrix_cls,
                model.hparams.symmetric_matrix,
                datamodule.sub_atomic_matrix,
                prediction=prediction)

        return app.run(host=host, port=port)

