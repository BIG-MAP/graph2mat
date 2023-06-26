import typer

from .siesta.main_cli import app as siesta_app
from .models.cli import app as models_app

app = typer.Typer(help="Command line interface for e3nn_matrix functionality.")

app.add_typer(models_app, name="models")
app.add_typer(siesta_app, name="siesta")

if __name__ == "__main__":
    app()