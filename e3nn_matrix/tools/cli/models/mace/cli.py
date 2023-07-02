import sys

import typer

app = typer.Typer(help="Interface to MACE models")

@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def main(ctx: typer.Context):
    """Main MACE model interface, using pytorch lightning CLI."""
    from e3nn_matrix.tools.lightning.models.mace import LitOrbitalMatrixMACE
    from e3nn_matrix.tools.lightning import OrbitalMatrixCLI, MatrixDataModule, SaveConfigSkipZTableCallback

    sys.argv = [ctx.command_path, *ctx.args]
    OrbitalMatrixCLI(LitOrbitalMatrixMACE, MatrixDataModule, save_config_callback=SaveConfigSkipZTableCallback)

if __name__ == "__main__":
    app()


