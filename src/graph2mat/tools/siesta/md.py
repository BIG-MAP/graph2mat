"""Tools to analyze how DM predictions can improve a MD run"""
from typing import Union, Iterable, Sequence, Callable, Optional, List

from pathlib import Path
import tempfile
import webbrowser

import sisl
import numpy as np
import pandas as pd

from jinja2 import Environment, FileSystemLoader


def _write_template(template_name, out, **kwargs):
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates")
    )
    template = environment.get_template(template_name)
    content = template.render(**kwargs)
    with open(out, "w") as f:
        f.write(content)


def setup(
    ml: Optional[str] = None,
    siesta: Optional[str] = None,
    inplace: bool = False,
    work_dir: Union[Path, None] = None,
    fdf_name: str = "graph2mat.fdf",
    lua_script: str = "graph2mat.lua",
    ml_model_name: str = "0",
    server_host: str = "localhost",
    server_port: int = 56000,
    server_api_endpoint: str = "api/extrapolation",
    store_interval: int = 0,
    store_dir: Path = "MD_steps",
    store_step_prefix: str = "",
    store_files: str = "*fdf *TSHS *TSDE *XV",
):
    """Sets up directories to run molecular dynamics using different DM initialization modes.

    This function will produce an fdf file (`fdf_name`) that you will need to include
    to your SIESTA main fdf input file.

    Notice that if you want to use the store functionality or machine learning predictions,
    you will need to use a SIESTA version that supports the lua language.

    History depth specifications
    ----------------------------
    Both `ml` and `siesta` arguments accept a history depth specification. This is a string that
    specifies the history depths, separated by commas, to be used with that method.
    E.g. "0" means that history depth 0 should be used, and "0,1"
    means that both history depths 0 and 1 should be used.

    Parameters
    ----------
    ml:
        Request DM predictions from an graph2mat server. **See history depth specifications in the help message.**

        Notice that you will need to run a server using the `graph2mat serve` command along with the SIESTA run.
    siesta:
        Use SIESTA extrapolation algorithm to calculate the DM using the previous steps.
        **See history depth specifications in the help message.**

        Notice that there are two special values of history depth:
            - 0: Initialize the DM from atomic coordinates.
            - 1: Use the DM from the last step.
    inplace :
        If True, the files are created in the current directory. Otherwise, a new directory
        is created for each initialization mode.
    work_dir :
        Directory where to setup things.
    fdf_name :
        Name of the fdf file that will be created.
    lua_script :
        Name of the lua script that will be (possibly) created. A lua file is only created
        if the store_interval is greater than 0 or the ml method is requested.
    ml_model_name :
        Name of the ML model to request predictions from the server. By default it is "0" which
        is the name of the default model in the server.
    server_host :
        Host where the graph2mat server will be running (for ML predictions).
    server_port :
        Port where the graph2mat server will be running (for ML predictions).
    server_api_endpoint :
        API endpoint to send prediction requests (for ML predictions).
    store_interval :
        Interval between two MD steps that are saved. If 0, no files are stored.
    store_dir :
        Directory where the step files are stored.
    store_step_prefix :
        Prefix to add to the step directories. This is useful if you want to run multiple
        dataset generating runs and store them in the same directory.
    store_files :
        String containing the files that should be kept in the dataset. This is passed
        directly to the `cp` shell command, so you can, for example, use wildcards.
    """
    work_dir = Path(work_dir) if work_dir is not None else Path.cwd()

    server_address = f"http://{server_host}:{server_port}/{server_api_endpoint}"

    lua_kwargs = {
        "include_store_code": store_interval > 0,
        "store_interval": store_interval,
        "store_dir": store_dir,
        "store_step_prefix": store_step_prefix,
        "store_files": store_files,
    }

    if siesta is not None:
        for history_len in siesta.split(","):
            history_len = int(history_len)

            if inplace:
                siesta_extrapolation_dir = work_dir
            else:
                siesta_extrapolation_dir = work_dir / f"siesta_{history_len}"
                siesta_extrapolation_dir.mkdir(exist_ok=True)

            _write_template(
                "fdf/dm_init_siesta_extrapolation.fdf"
                if history_len > 0
                else "fdf/dm_init_atomic.fdf",
                siesta_extrapolation_dir / fdf_name,
                history_depth=history_len,
                extra_string=f"Lua.Script {lua_script}" if store_interval > 0 else "",
            )

            if store_interval > 0:
                _write_template(
                    "lua/graph2mat.lua",
                    siesta_extrapolation_dir / lua_script,
                    **lua_kwargs,
                )

    if ml is not None:
        for history_len in ml.split(","):
            history_len = int(history_len)

            if inplace:
                ml_dir = work_dir
            else:
                ml_dir = work_dir / f"ml_{history_len}"
                ml_dir.mkdir(exist_ok=True)

            _write_template(
                "fdf/dm_init_ml.fdf", ml_dir / fdf_name, lua_script=lua_script
            )

            lua_kwargs.update(
                {
                    "include_server_code": True,
                    "server_address": server_address,
                    "main_fdf": "RUN.fdf",
                    "ml_model_name": ml_model_name,
                    "work_dir": f'"{ml_dir}"',
                    "history_len": history_len,
                }
            )

            _write_template("lua/graph2mat.lua", ml_dir / lua_script, **lua_kwargs)


def setup_store(
    dir: Path = "MD_steps",
    step_prefix: str = "",
    interval: int = 1,
    files: str = "*fdf *TSHS *TSDE *XV",
    out: Path = "md_store.lua",
):
    """Prepares a lua script that, if included in a SIESTA run, generates a dataset.

    The resulting script is meant to be passed to SIESTA using the `Lua.Script` flag.

    Parameters
    ----------
    dir : Union[Path, str], optional
        Path to the directory where the dataset will be created/stored. Note that this
        is relative to wherever you run SIESTA, so you might want to use an absolute path.
    step_prefix : str, optional
        Prefix to add to the step directories. This is useful if you want to run multiple
        dataset generating runs and store them in the same directory.
    interval : int, optional
        Interval between two steps that are stored in the dataset.
    files : str, optional
        String containing the files that should be kept in the dataset. This is passed
        directly to the `cp` shell command, so you can, for example, use wildcards.
    lua_script : Union[Path, str], optional
        Path where the resulting lua script should be stored.
    """

    lua_kwargs = {
        "include_store_code": True,
        "store_interval": interval,
        "store_dir": dir,
        "store_step_prefix": step_prefix,
        "store_files": files,
        "include_server_code": False,
    }

    _write_template("lua/graph2mat.lua", out, **lua_kwargs)


def md_guess_performance_dataframe(out_file: Union[str, Path]) -> pd.DataFrame:
    """Returns a dataframe describing how close the first SCF step is to the converged properties.

    The resulting dataframe contains information for each MD step.

    Parameters
    ----------
    out_file : Union[str, Path]
        Path to SIESTA's output file.
    """
    out_sile = sisl.get_sile(out_file, cls=sisl.io.siesta.stdoutSileSiesta)

    # Read the first and last iteration of every scf loop,
    # we don't care about the iterations in the middle (for now)
    md_scf_first = out_sile.read_scf[1:](iscf=1, as_dataframe=True)
    md_scf_conv = out_sile.read_scf[1:](iscf=-1, as_dataframe=True)

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
    agg: Union[Sequence[Union[Callable, str]], None] = ("mean", "min", "max", "std"),
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
    names = []
    for out_file in out_files:
        out_file = Path(out_file)
        df = md_guess_performance_dataframe(out_file)
        name = out_file.parent.name
        if name in names:
            name = str(out_file.parent)
        names.append(name)
        df["Run name"] = name
        dfs.append(df)

    # Concatenate all the dataframes
    df = pd.concat(dfs)

    # Aggregate MD step data if requested
    if agg is not None:
        df = df.groupby("Run name").agg(agg)

    return df


def visualize_performance_table(
    out_files: List[Path],
    agg: List[str] = ["mean", "min", "max", "std"],
    precision: int = 3,
    notebook: bool = False,
    save: Union[str, None] = None,
):
    """Styles the performance dataframe so to help visualization.

    For now, this function must be run in an IPython notebook.

    Parameters
    ----------
    out_files :
        List of SIESTA output files of the different MDs.
    agg :
        List of aggregation functions to aggregate MD steps.
        They are obtained from the numpy module.
        Note that the MD steps are aggregated separately for each run.
    precision:
        The number of decimal places to show in the table.
    notebook :
        If True, the table is displayed in the IPython notebook.
        Otherwise, it is displayed in the browser.
        If save is provided, this parameter is ignored.
    save :
        If provided, the HTML table is saved to the specified path.

        If this is a path to a csv file, the dataframe is saved to a csv file
        instead.
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
    if save is not None:
        df.columns = ["_".join(col) for col in df.columns]
        if Path(save).suffix == ".csv":
            return df.to_csv(save)
        return styler.to_html(save)
    elif notebook:
        return styler
    else:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)

        # Open the file for writing.
        with open(tmp.name, "w") as f:
            f.write(styler.to_html())

        return webbrowser.open(tmp.name)
