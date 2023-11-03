from typing import Optional
from typing_extensions import Annotated

import typer

app = typer.Typer(help="Utilities for molecular dynamics runs.")


@app.command()
def analyze(
    precision: Annotated[
        int, typer.Option(help="Number of decimal places to show in the table.")
    ] = 3,
    save_path: Annotated[
        Optional[str],
        typer.Option(
            help="Path to save the HTML table to. If not provided, the table is displayed in the browser."
        ),
    ] = None,
    *out_files: Annotated[
        str, typer.Argument(help="Paths to the output files of the MD runs.")
    ],
):
    """Analyzes the output of a MD run and generates a table with the performance of the run."""
    from e3nn_matrix.tools.siesta.md import visualize_performance_table

    visualize_performance_table(out_files, precision=precision, save_path=save_path)


@app.command()
def prepare_gen_dataset(
    dataset_dir: Annotated[
        str,
        typer.Option(
            help="Path to the directory where the dataset will be created/stored."
            " Note that this is relative to wherever you run SIESTA, so you might want to use an absolute path."
        ),
    ] = "MD_dataset",
    stepdir_prefix: Annotated[
        str,
        typer.Option(
            help="Prefix to add to the step directories. This is useful if you want to run multiple"
            " dataset generating runs and store them in the same directory."
        ),
    ] = "",
    store_interval: Annotated[
        int,
        typer.Option(help="Interval between two steps that are stored in the dataset."),
    ] = 1,
    files_to_keep: Annotated[
        str,
        typer.Option(
            help="String containing the files that should be kept in the dataset. This is passed "
            "directly to the `cp` shell command, so you can, for example, use wildcards. If you use wildcards, make sure to wrap the "
            "input in single quotes (') to avoid the shell from expanding"
        ),
    ] = "*fdf *TSHS *TSDE *XV",
    out: Annotated[
        str, typer.Option(help="Path where the resulting lua script should be stored.")
    ] = "gen_dataset.lua",
):
    """Prepares a lua script that, if included in a SIESTA run, generates a dataset.

    The resulting script is meant to be passed to SIESTA using the `Lua.Script` flag.
    """
    from e3nn_matrix.tools.siesta.md import prepare_gen_dataset

    prepare_gen_dataset(
        dataset_dir=dataset_dir,
        stepdir_prefix=stepdir_prefix,
        store_interval=store_interval,
        files_to_keep=files_to_keep,
        out=out,
    )


if __name__ == "__main__":
    app()
