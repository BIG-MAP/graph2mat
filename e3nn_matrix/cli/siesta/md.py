from typing import Optional
from typing_extensions import Annotated

import typer

app = typer.Typer(help="Utilities for molecular dynamics runs.")

@app.command()
def analyze(
    precision: Annotated[int, typer.Option(help="Number of decimal places to show in the table.")] = 3, 
    save_path: Annotated[Optional[str], typer.Option(help="Path to save the HTML table to. If not provided, the table is displayed in the browser.")] = None, 
    *out_files: Annotated[str, typer.Argument(help="Paths to the output files of the MD runs.")]
):
    """Analyzes the output of a MD run and generates a table with the performance of the run."""
    from e3nn_matrix.scripts.siesta.analyze_MD import visualize_performance_table
    
    visualize_performance_table(out_files, precision=precision, save_path=save_path)

if __name__ == "__main__":
    app()