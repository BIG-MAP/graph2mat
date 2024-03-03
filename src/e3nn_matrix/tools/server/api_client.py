"""Simple HTTP client to interact with the server."""

from typing import Union, List, Type, Optional

import tempfile
import os
import urllib.parse

import requests
from pathlib import Path

import sisl


class ServerClient:
    """Client to interact easily with the e3nn server.

    Parameters
    ----------
    url : str or None, optional
        Root url where the server is running.

        If it is set to None, the environment variable E3MAT_SERVER_URL
        will be used if present.

        If it is set to None and the environment variable is not present,
        the url will be constructed from the values of ``host`` and ``port``.
    host : str or None, optional
        Host where the server is running.

        If it is set to None, the environment variable E3MAT_SERVER_HOST
        will be used if present, otherwise the default value of ``host`` will
        be used.
    port : int, optional
        Port where the server is listening.

        If it is set to None, the environment variable E3MAT_SERVER_PORT
        will be used if present, otherwise the default value of ``port`` will
        be used.
    """

    host: Union[str, None]
    port: Union[int, None]
    root_url: str

    def __init__(
        self,
        url: Optional[str] = None,
        host: Optional[str] = "localhost",
        port: Optional[int] = 56000,
    ):
        url = url or os.environ.get("E3MAT_SERVER_URL", None)

        if url is None:
            self.host = host or os.environ.get("E3MAT_SERVER_HOST", "localhost")
            self.port = port or int(os.environ.get("E3MAT_SERVER_PORT", 56000))

            self.root_url = f"http://{self.host}:{self.port}"
        else:
            parsed_url = urllib.parse.urlparse(url)

            self.host = parsed_url.hostname
            self.port = parsed_url.port
            self.root_url = url

        self.api_url = f"{self.root_url}/api"

    def avail_models(self) -> List[str]:
        """Returns the models that are available in the server."""
        response = requests.get(f"{self.api_url}/avail_models")
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        geometry: Union[str, Path, sisl.Geometry],
        output: Union[str, Path, Type[sisl.SparseOrbital]],
        model: str,
        local: bool = False,
    ) -> Union[Path, sisl.SparseOrbital]:
        """Predicts the matrix for a given geometry.

        Parameters
        ----------
        geometry : Union[str, sisl.Geometry]
            Either the path to the geometry file or the geometry itself.
        output : Union[str, Type[sisl.SparseOrbital]]
            Either the path to the file where the prediction should be saved
            or the type of the object to be returned.
        model : str
            Name of the model to use for the prediction. This model must be available
            in the server. You can check the available models with the `avail_models` method.
        local : bool
            Whether the paths (if given) are in the local filesystem.
        """
        # We may need temporal files to transmit the geometry and the output
        input_tmp_file = None
        output_tmp_file = None

        # Regardless of what the argument 'geometry' is, we will pass a file
        # to the server.
        # If a geometry is provided, we store it to a temporary file.
        if isinstance(geometry, sisl.Geometry):
            input_tmp_file = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
            geometry_path = Path(input_tmp_file.name).absolute()
            geometry = geometry.write(geometry_path)
        else:
            geometry_path = Path(geometry).absolute()

        # Regardless of what the argument 'output' is, the server0's output will
        # be written to a file.
        # If the user wants a sisl object, we will just tell the server to output
        # to a temporary file, which we will then parse.
        if isinstance(output, type) and issubclass(output, sisl.SparseOrbital):
            suffix = {
                sisl.DensityMatrix: ".DM",
                sisl.Hamiltonian: ".TSHS",
                sisl.EnergyDensityMatrix: ".EDM",
            }[output]

            output_tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            output_path = Path(output_tmp_file.name).absolute()
        else:
            output_path = Path(output).absolute()

        # Now we can make the request to the server.
        if local:
            # We ask the server to read and write to its local filesystem.
            response = requests.get(
                f"{self.api_url}/models/{model}/local_write_predict",
                params={
                    "geometry_path": str(geometry_path),
                    "output_path": str(output_path),
                },
            )

            response.raise_for_status()
        else:
            # We send the geometry file from our filesystem to the server.
            # We will also receive a file from the server (binary content).
            files = {"geometry_file": open(geometry_path, "rb")}

            response = requests.post(
                f"{self.api_url}/models/{model}/predict", files=files
            )

            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

        # Remove the temporal files if they were created.
        if input_tmp_file is not None:
            Path(input_tmp_file.name).unlink()

        if output_tmp_file is not None:
            # In the case that we wanted a sisl obect as output, we parse the
            # file that the server sent us, using the read method of the sisl
            # object.
            returns = output.read(output_path)
            Path(output_path).unlink()
        else:
            returns = output_path

        return returns
