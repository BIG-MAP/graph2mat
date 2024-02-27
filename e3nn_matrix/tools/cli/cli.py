import typer

from .siesta.main_cli import app as siesta_app
from .models.cli import app as models_app
from .serve import app as serve_app
from .request import app as request_app

app = typer.Typer(
    help="Command line interface for e3nn_matrix functionality.",
    pretty_exceptions_show_locals=False,
    rich_markup_mode="markdown",
)

app.add_typer(models_app, name="models")
app.add_typer(siesta_app, name="siesta")
app.add_typer(serve_app, name="serve")
app.add_typer(request_app, name="request")

if __name__ == "__main__":
    app()
