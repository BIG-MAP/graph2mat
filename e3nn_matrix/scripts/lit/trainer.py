from typing import Optional

from pytorch_lightning import Trainer, LightningModule, LightningDataModule
import torch

from e3nn_matrix.torch.data import OrbitalMatrixData

class MatrixTrainer(Trainer):
    """Pytorch lightning trainer with some extra functionality for matrices."""

    def serve(
        self,
        model: LightningModule,
        ckpt_path: str,
        output_file: str = "predicted.DM",
        datamodule: Optional[LightningDataModule] = None,
        host="localhost",
        port: int=56000,
    ):
        r"""
        Runs a server that provides predictions with a trained model.

        The implementation does not follow the regular lightning loop,
        so callbacks etc. will not work.

        Parameters
        ----------
        model : LightningModule
            Model definition
        ckpt_path : str
            Path to the checkpoint file.
        output_file : str, optional
            Name of the output file where the predictions will be written, 
            by default "predicted.DM", which will be written in the same directory
            as the input file.
        datamodule : Optional[LightningDataModule], optional
            Needed to determine the matrix type, by default None.
        host : str, optional
            Hostname to use for the server, by default "localhost"
        port : int, optional
            Port to use for the server.
        """
        from ..server.predictions_server import predictions_server_app

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

        # Define the function that will be evaluated to predict the output.
        # It receives an empty matrix_data object with the information of the
        # structure only.
        def predict(matrix_data: OrbitalMatrixData):
            with torch.no_grad():
                prediction = model(matrix_data)
            return prediction
        
        # Create the server app
        app = predictions_server_app(
            prediction_function=predict,
            z_table=model.z_table,
            sub_atomic_matrix=datamodule.sub_atomic_matrix,
            symmetric_matrix=model.hparams.symmetric_matrix,
            out_matrix=out_matrix,
            output_file_name=output_file,
        )

        # And run it.
        return app.run(host=host, port=port)

