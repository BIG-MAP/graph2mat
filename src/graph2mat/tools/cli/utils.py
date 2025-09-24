import typer
from typing_extensions import Annotated

app = typer.Typer(help="Utilities to easily interact with the e3nn server.")


@app.command()
def precompute_rtps(
    max_l: Annotated[
        int,
        typer.Option(
            help="Maximum irreps l to store",
        ),
    ] = 4,
):
    """Store the precomputed change of basis and irreps_out of reduced tensor products.

    This is done to avoid the very expensive initialization of e3nn.o3.ReducedTensorProducts.
    """
    from graph2mat.bindings.e3nn import store_precomputed_rtps

    store_precomputed_rtps(max_l=max_l)


if __name__ == "__main__":
    app()
