from typing import Optional, List, Union
from typing_extensions import Annotated

from pathlib import Path

import typer

from graph2mat.tools.cli._typer import annotate_typer
from graph2mat.tools.siesta.md import (
    visualize_performance_table,
    setup,
    setup_store,
)

app = typer.Typer(help="Utilities for molecular dynamics runs.")

app.command("analyze")(annotate_typer(visualize_performance_table))

app.command()(annotate_typer(setup))

app.command()(annotate_typer(setup_store))

if __name__ == "__main__":
    app()
