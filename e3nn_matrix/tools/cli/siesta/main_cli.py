import typer

from .md import app as md_app

app = typer.Typer(
    help="Set of utilities to interface the machine learning models with SIESTA."
)

app.add_typer(md_app, name="md")
