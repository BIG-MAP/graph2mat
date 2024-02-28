import typer

from .mace.cli import app as mace_app

app = typer.Typer(
    help="""
Interface to ML models that have been adapted to use e3nn_matrix.
For each model, we just defer to the Pytorch Lightning CLI, so if
you do --help, you will see the Pytorch Lightning CLI help.

NOTE: We did a
"""
)

app.add_typer(mace_app, name="mace")

if __name__ == "__main__":
    app()
