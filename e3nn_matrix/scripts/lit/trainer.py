from pathlib import Path
from typing import Optional, Union
import contextlib
import copy
import pickle

from pytorch_lightning import Trainer, LightningModule, LightningDataModule
import numpy as np
import torch
import sisl

from e3nn_matrix.data.configuration import load_orbital_config_from_run
from e3nn_matrix.torch.data import OrbitalMatrixData
from e3nn_matrix.data.periodic_table import AtomicTableWithEdges
#from e3nn_matrix.bindings.mace.lit import LitOrbitalMatrixMACE

def _write_matrix_data_to_file(filename: Union[Path,str], matrix_data: OrbitalMatrixData, z_table: AtomicTableWithEdges, matrix_cls, symmetric_matrix: bool, sub_atomic_matrix: bool, prediction=None):
    matrix_data_cp = copy.deepcopy(matrix_data)

    if prediction is not None:
        matrix_data_cp.atom_labels = prediction['node_labels']
        matrix_data_cp.edge_labels = prediction['edge_labels']

    if len(matrix_data_cp.atom_labels.shape) > 1:
        # More than one element per node and edge
        assert matrix_data_cp.edge_labels.shape[1:] == matrix_data_cp.atom_labels.shape[1:]
        flat_atom = matrix_data_cp.atom_labels.flatten(start_dim=1, end_dim=-1)
        flat_edge = matrix_data_cp.edge_labels.flatten(start_dim=1, end_dim=-1)
        sparse_matrices = []
        # Create a sparse matrix for each label
        for atom, edge in zip(torch.unbind(flat_atom, dim=1), torch.unbind(flat_edge, dim=1)):
            matrix_data_cp.atom_labels = atom
            matrix_data_cp.edge_labels = edge
            sparse_orbital_matrix = matrix_data_cp.to_sparse_orbital_matrix(z_table, matrix_cls, symmetric_matrix, sub_atomic_matrix)
            geometry = sparse_orbital_matrix.geometry
            sparse_matrices.append(sparse_orbital_matrix.tocsr())
        # Concatenate the sparse matrices to a single sparse NxNxM matrix
        sparse_orbital_matrix = matrix_cls.fromsp(geometry, sparse_matrices)
    else:
        sparse_orbital_matrix = matrix_data_cp.to_sparse_orbital_matrix(z_table, matrix_cls, symmetric_matrix, sub_atomic_matrix)

    # And write the matrix to it.
    if Path(filename).suffix == ".pkl":
        with open(filename, "wb") as f:
            pickle.dump(sparse_orbital_matrix, f)
    else:
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
            out_force_file = args.get('output_force', None)
            out_grad_file = args.get('output_grad', None)

            runfile = Path(input_file)
            config = load_orbital_config_from_run(runfile, out_matrix=None)

            matrix_data = OrbitalMatrixData.from_config(
                config,
                z_table=model.z_table,
                sub_atomic_matrix=datamodule.sub_atomic_matrix,
                symmetric_matrix=model.hparams.symmetric_matrix,
            )

            if out_force_file is None and out_grad_file is None:
                # No gradients are needed, calculate with no_grad contextmanager
                grad_ctxt = torch.no_grad()
            else:
                grad_ctxt = contextlib.nullcontext()

            # Load hamiltonian matrix if available
            if out_force_file is not None:
                config = load_orbital_config_from_run(runfile, out_matrix="hamiltonian")
                hamiltonian_matrix_data = OrbitalMatrixData.from_config(
                    config,
                    z_table=model.z_table,
                    sub_atomic_matrix=False,
                    symmetric_matrix=False,
                )
            else:
                hamiltonian_matrix_data = None

            # Run model
            with grad_ctxt:
                prediction = model(matrix_data,
                    calculate_forces=(out_force_file is not None),
                    calculate_grads=(out_grad_file is not None),
                    hamiltonian=hamiltonian_matrix_data
                )

            # Write forces to file
            if out_force_file is not None:
                # Convert to numpy
                out_forces = prediction["forces"].numpy(force=True)
                path = matrix_data.metadata["path"]
                output_force_path = path.parent / out_force_file
                np.savetxt(output_force_path, out_forces)

            # Write gradients to file
            if out_grad_file is not None:
                pred_grad = {"node_labels": prediction["node_grads"],
                        "edge_labels": prediction["edge_grads"]
                        }

                path = matrix_data.metadata["path"]
                output_grad_path = path.parent / out_grad_file
                _write_matrix_data_to_file(
                    output_grad_path,
                    matrix_data,
                    model.z_table,
                    sisl.SparseOrbital,
                    symmetric_matrix=model.hparams.symmetric_matrix,
                    sub_atomic_matrix=False,
                    prediction=pred_grad)



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

