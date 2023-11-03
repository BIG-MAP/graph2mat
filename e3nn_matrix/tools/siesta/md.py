"""Tools to analyze how DM predictions can improve a MD run"""
from typing import Union, Iterable, Sequence, Callable

from pathlib import Path
import tempfile
import webbrowser

import sisl
import numpy as np
import pandas as pd


def prepare_gen_dataset(
    dataset_dir: Union[Path, str] = "MD_dataset",
    stepdir_prefix: str = "",
    store_interval: int = 1,
    files_to_keep: str = "*fdf *TSHS *TSDE *XV",
    out: Union[Path, str] = "gen_dataset.lua",
):
    """Prepares a lua script that, if included in a SIESTA run, generates a dataset.

    The resulting script is meant to be passed to SIESTA using the `Lua.Script` flag.

    Parameters
    ----------
    dataset_dir : Union[Path, str], optional
        Path to the directory where the dataset will be created/stored. Note that this
        is relative to wherever you run SIESTA, so you might want to use an absolute path.
    stepdir_prefix : str, optional
        Prefix to add to the step directories. This is useful if you want to run multiple
        dataset generating runs and store them in the same directory.
    store_interval : int, optional
        Interval between two steps that are stored in the dataset.
    files_to_keep : str, optional
        String containing the files that should be kept in the dataset. This is passed
        directly to the `cp` shell command, so you can, for example, use wildcards.
    out : Union[Path, str], optional
        Path where the resulting lua script should be stored.
    """
    from jinja2 import Environment, FileSystemLoader

    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates")
    )
    template = environment.get_template("gen_dataset.lua")

    # Render the template
    content = template.render(
        dataset_dir=str(dataset_dir),
        stepdir_prefix=stepdir_prefix,
        store_interval=store_interval,
        files_to_keep=files_to_keep,
    )

    # Write the generated lua file
    with open(out, "w") as f:
        f.write(content)


def md_guess_performance_dataframe(out_file: Union[str, Path]) -> pd.DataFrame:
    """Returns a dataframe describing how close the first SCF step is to the converged properties.

    The resulting dataframe contains information for each MD step.

    Parameters
    ----------
    out_file : Union[str, Path]
        Path to SIESTA's output file.
    """
    out_sile = sisl.get_sile(out_file, cls=sisl.io.siesta.outSileSiesta)

    # Read the first and last iteration of every scf loop,
    # we don't care about the iterations in the middle (for now)
    md_scf_first = out_sile.read_scf(iscf=1, as_dataframe=True)[1:]
    md_scf_conv = out_sile.read_scf(iscf=-1, as_dataframe=True)[1:]

    df = pd.concat(
        [
            md_scf_conv.iscf,
            *[md_scf_first[k] for k in ("dDmax", "dHmax")],
            *[abs(md_scf_conv[k] - md_scf_first[k]) for k in ("E_KS", "Ef")],
        ],
        axis=1,
    )

    return df.rename(
        columns={
            "iscf": "SCF steps",
            "dDmax": "First dDmax",
            "dHmax": "First dHmax",
            "E_KS": "E_KS error",
            "Ef": "Ef error",
        }
    )


def md_guess_performance_dataframe_multi(
    out_files: Iterable[Union[str, Path]],
    agg: Union[Sequence[Callable], None] = (np.mean, min, max, np.std),
) -> pd.DataFrame:
    """Returns a dataframe describing how close the first SCF step is to the converged properties.

    Same as md_guess_performance_dataframe, but adds an extra column that specifies the run name.

    Parameters
    ----------
    out_files : Iterable[Union[str, Path]]
        Iterable that returns paths to SIESTA's output files of the different MDs.
    agg : (Iterable of functions) or None, optional
        If it is None, the df contains information for each MD step.
        Otherwise, the MD steps are aggregated using the provided functions.
        Note that the MD steps are aggregated separately for each run.
    """
    dfs = []
    # Loop through output files and get the dataframe for each of them,
    # adding the run name.
    for out_file in out_files:
        out_file = Path(out_file)
        df = md_guess_performance_dataframe(out_file)
        df["Run name"] = out_file.parent.name
        dfs.append(df)

    # Concatenate all the dataframes
    df = pd.concat(dfs)

    # Aggregate MD step data if requested
    if agg is not None:
        df = df.groupby("Run name").agg(agg)

    return df


def visualize_performance_table(
    out_files: Iterable[Union[str, Path]],
    agg: Sequence[Callable] = (np.mean, min, max, np.std),
    precision: int = 3,
    notebook: bool = False,
    save_path: Union[str, None] = None,
):
    """Styles the performance dataframe so to help visualization.

    For now, this function must be run in an IPython notebook.

    Parameters
    ----------
    out_files : Iterable[Union[str, Path]]
        Iterable that returns paths to SIESTA's output files of the different MDs.
    agg : Iterable of functions, optional
        MD steps are aggregated using the provided functions.
        Note that the MD steps are aggregated separately for each run.
    precision: int, optional
        The number of decimal places to show in the table.
    notebook : bool, optional
        If True, the table is displayed in the IPython notebook.
        Otherwise, it is displayed in the browser.
        If save_path is provided, this parameter is ignored.
    save_path : Union[str, None], optional
        If provided, the HTML table is saved to the specified path.
    """
    # Get the performance dataframe
    df = md_guess_performance_dataframe_multi(out_files, agg=agg)

    # Define the function that will style the dataframe table.
    def _style_performance_table(styler):
        def highlight_max(s, props=""):
            return np.where(s == np.nanmax(s.values), props, "")

        def highlight_min(s, props=""):
            return np.where(s == np.nanmin(s.values), props, "")

        def property_border(s):
            return [
                "border-bottom: dashed 2px gray" if (x + 1) % 4 == 0 else ""
                for x in range(len(s))
            ]

        # Title of the table
        styler.set_caption("MD comparison")
        # Global styles
        styler.set_properties(**{"text-align": "center", "border": "solid black 1px"})

        # Coloring of background according to values
        styler.background_gradient(axis=1, cmap="RdYlGn_r")

        # Highlighting of max and min values
        styler.apply(highlight_max, props="font-weight:bold;", axis=1)
        styler.apply(highlight_min, props="font-weight:bold;", axis=1)

        # Set separator between properties
        styler.apply(property_border)

        # Specify precision for value printing
        styler.format(precision=precision)
        return styler

    # Get the styler object and apply the style function
    styler = df.T.style.pipe(_style_performance_table)

    # Show or save the table, depending on what the user requested
    if save_path is not None:
        return styler.to_html(save_path)
    elif notebook:
        return styler
    else:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)

        # Open the file for writing.
        with open(tmp.name, "w") as f:
            f.write(styler.to_html())

        return webbrowser.open(tmp.name)
