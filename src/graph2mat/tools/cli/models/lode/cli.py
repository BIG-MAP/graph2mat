import sys

import typer

app = typer.Typer(help="Interface to LODE models")


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def main(ctx: typer.Context):
    """Main LODE model interface, using pytorch lightning CLI."""
    from graph2mat.tools.lightning.models.lode import LitLODEMatrixModel
    from graph2mat.tools.lightning import (
        OrbitalMatrixCLI,
        MatrixDataModule,
        SaveConfigSkipBasisTableCallback,
    )

    sys.argv = [ctx.command_path, *ctx.args]
    OrbitalMatrixCLI(
        LitLODEMatrixModel,
        MatrixDataModule,
        save_config_callback=SaveConfigSkipBasisTableCallback,
    )


if __name__ == "__main__":
    app()
