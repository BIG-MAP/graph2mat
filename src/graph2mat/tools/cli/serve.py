from typing import List
from typing_extensions import Annotated

import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
    models: Annotated[
        List[str],
        typer.Argument(
            help="""List of models to load. Each model can be provided either as a .ckpt file, a .yaml specification
        or a directory that contains a 'spec.yaml' file.
        Regardless of what you provide, you can specify the name of the model like 'model_name:file.ckpt',
        that is, separated from the file name using a semicolon."""
        ),
    ],
    host: Annotated[
        str, typer.Option(help="Host to launch the server.", envvar="E3MAT_SERVER_HOST")
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            help="Port where the server should listen.", envvar="E3MAT_SERVER_PORT"
        ),
    ] = 56000,
    cpu: Annotated[
        bool,
        typer.Option(
            help="Load parameters in the CPU regardless of whether they were in the GPU."
        ),
    ] = True,
    local: Annotated[
        bool,
        typer.Option(
            help="If True, the server allows the user to ask for changes in the local file system."
        ),
    ] = False,
):
    import uvicorn

    from graph2mat.tools.server import create_server_app_from_filesystem

    # Sanitize the ckpt files, building a dictionary with names and files.
    ckpt_files_dict = {}
    for i, model_file in enumerate(models):
        splitted = model_file.split(":")

        if len(splitted) == 2:
            model_name, model_file = splitted
        else:
            model_name = str(i)

        ckpt_files_dict[model_name] = model_file

    # Then build the app
    fastapi_app = create_server_app_from_filesystem(
        ckpt_files_dict, cpu=cpu, local=local
    )

    # And launch it.
    uvicorn.run(fastapi_app, host=host, port=port)


if __name__ == "__main__":
    app()
