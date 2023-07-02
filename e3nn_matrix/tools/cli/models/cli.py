import typer

from .mace.cli import app as mace_app

app = typer.Typer(help="Interface to ML models that have been adapted to use e3nn_matrix")

app.add_typer(mace_app, name="mace")

if __name__ == "__main__":
    app()