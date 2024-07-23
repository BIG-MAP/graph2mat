from typing import Optional
from typing_extensions import Annotated
from requests.exceptions import HTTPError
import warnings

import typer

app = typer.Typer(help="Utilities to easily interact with the e3nn server.")


@app.command()
def avail_models(
    host: Annotated[
        str,
        typer.Option(
            help="Host where the server is running.", envvar="E3MAT_SERVER_HOST"
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            help="Port where the server is listening.", envvar="E3MAT_SERVER_PORT"
        ),
    ] = 56000,
    url: Annotated[
        Optional[str],
        typer.Option(help="URL of the server.", envvar="E3MAT_SERVER_URL"),
    ] = None,
):
    """Shows the models available in the server."""
    from graph2mat.tools.server import ServerClient

    client = ServerClient(host=host, port=port, url=url)

    print(client.avail_models())


@app.command()
def predict(
    geometry: Annotated[
        str,
        typer.Argument(
            help="Path to the geometry file for which the prediction is desired."
        ),
    ],
    output: Annotated[
        str,
        typer.Argument(help="Path to the file where the prediction should be saved."),
    ],
    model: Annotated[
        str, typer.Option(help="Name of the model to use for the prediction.")
    ],
    local: Annotated[
        bool, typer.Option(help="Whether the paths are in the local filesystem.")
    ] = False,
    host: Annotated[
        str,
        typer.Option(
            help="Host where the server is running.", envvar="E3MAT_SERVER_HOST"
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            help="Port where the server is listening.", envvar="E3MAT_SERVER_PORT"
        ),
    ] = 56000,
    url: Annotated[
        Optional[str],
        typer.Option(help="URL of the server.", envvar="E3MAT_SERVER_URL"),
    ] = None,
):
    """Predict the matrix for a given geometry."""
    # Import the server client class, which will be used to interact with the server.
    from graph2mat.tools.server import ServerClient

    client = ServerClient(host=host, port=port, url=url)

    try:
        client.predict(geometry=geometry, output=output, model=model, local=local)
    except HTTPError as e:
        if e.response.status_code == 422:
            avail_models = client.avail_models()
            if model not in avail_models:
                raise ValueError(
                    f"Model '{model}' not available in the server. Available models are: {avail_models}"
                )

        raise e


if __name__ == "__main__":
    app()
