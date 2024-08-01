"""Implements a fastapi server API that accepts requests to predict matrices."""


from typing import Callable, Dict, Optional, TypedDict, List, Union

from enum import Enum
import dataclasses

from pathlib import Path
import tempfile
from io import StringIO
from functools import partial

import sisl
import numpy as np
import pandas as pd

import torch

from graph2mat import MatrixDataProcessor, BasisMatrixData
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.bindings.torch.load import load_from_lit_ckpt

from graph2mat.tools.server.extrapolation import create_extrapolation_app

try:
    from fastapi import (
        FastAPI,
        UploadFile,
        BackgroundTasks,
        HTTPException,
        Request,
        Form,
    )
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
    from fastapi.templating import Jinja2Templates
    from fastapi.staticfiles import StaticFiles
except ImportError as e:

    class FastAPI:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "You need to install fastapi to initialize a server."
            ) from e

        def get(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        post = get


class Files(TypedDict):
    ckpt: Union[Path, str]
    sample_metrics: Union[Path, str]
    basis: Union[Path, str]
    structs: Union[Path, str]


class ModelSpecification(TypedDict):
    prediction_function: Callable[[BasisMatrixData], Dict[str, np.ndarray]]
    data_processor: MatrixDataProcessor
    description: str
    authors: List[str]
    files: Files
    root_dir: Path
    test_metrics_summary: str


def create_server_app(
    models: Dict[str, ModelSpecification],
    local: bool = False,
) -> FastAPI:
    """Creates a flask app to listen to requests and predict matrices.

    The app is to be ran with uvicorn. For example:

    >>> import uvicorn
    >>> models = {}
    >>> app = server_app(models)
    >>> uvicorn.run(app, host="localhost", port=56000)

    Parameters
    ----------
    models: Dict[str, ModelSpecification]
        A dictionary with the models to be used. The keys are the names of the models, and the values
        are dictionaries with everything that we need/know about the model.
    local: bool, optional
        If True, the server allows the user to ask for changes in the local file system.
    """

    app = FastAPI(
        title="E3nn_matrix server",
    )

    api = FastAPI(
        title="E3nn_matrix server API",
        description="""API that allows the interaction between the models trained with e3nn_matrix
        and the codes that use their predictions.""",
    )

    app.mount("/api", api)

    # Valid model names.
    ModelName = Enum("ModelName", {k: k for k in models}, type=str)

    for model in models.values():
        if "test_metrics_summary" in model:
            continue

        model["test_metrics_summary"] = ""

        test_metrics_file = model["files"].get("test_metrics_summary")
        if test_metrics_file is None:
            continue

        test_metrics_file = Path(test_metrics_file)
        if test_metrics_file.suffix != ".csv":
            continue

        if not test_metrics_file.is_absolute():
            test_metrics_file = Path(model["root_dir"]) / test_metrics_file

        if test_metrics_file.exists():
            try:
                metrics = pd.read_csv(test_metrics_file, index_col=0)
            except:
                continue

            model["test_metrics_summary"] = metrics.to_html()

    class ModelFile(str, Enum):
        """Valid model files."""

        ckpt = "ckpt"
        test_metrics_summary = "test_metrics_summary"
        sample_metrics = "sample_metrics"
        basis = "basis"
        structs = "structs"

    class ModelAction(str, Enum):
        predict = "predict"
        test = "test"

    actions = {
        ModelAction.predict: {
            "name": ModelAction.predict.value,
            "short_help": "When predicting, you send a structure and you receive the matrix that the model predicts.",
        },
        ModelAction.test: {
            "name": ModelAction.test.value,
            "short_help": "When testing, you send both a structure and the expected output, and you get metrics on how well the model performs.",
        },
    }

    class Tab(str, Enum):
        """Valid tabs in the frontend."""

        models = "models"
        about = "about"

    tabs = {
        Tab.models: {
            "name": Tab.models.value,
        },
        Tab.about: {
            "name": Tab.about.value,
        },
    }

    @api.get("/avail_models")
    def return_available_models() -> List[str]:
        return list(models.keys())

    @api.get("/models/{model_name}/info")
    async def model_info(model_name: ModelName):
        """Returns information about a model."""
        model = models[model_name.value]

        info = {
            k: v
            for k, v in dataclasses.asdict(model["data_processor"]).items()
            if k not in ["z_table"]
        }

        info["description"] = model["description"]

        return info

    @api.get("/models/{model_name}/avail_info")
    async def model_avail_info(model_name: ModelName) -> List[str]:
        """Returns the list of keys that are available in the model's info."""
        model = models[model_name.value]

        return list(model)

    @api.get("/models/{model_name}/files/{file_name}", response_class=FileResponse)
    async def model_files(model_name: ModelName, file_name: ModelFile):
        """Download files related to the model"""
        model = models[model_name.value]
        filename = file_name.value

        file_path = Path(model["files"][filename])

        if not file_path.is_absolute():
            file_path = Path(model.get("root_dir", ".")) / file_path

        if filename in ["ckpt"]:
            response = FileResponse(
                file_path,
                media_type="application/octet-stream",
                filename=file_path.name,
            )
        else:
            response = str(file_path)

        return response

    @api.get("/models/{model_name}/files")
    async def model_files_info(model_name: ModelName):
        """Information about the files available for the model."""
        model = models[model_name.value]

        return {
            "available_files": list(model["files"].keys()),
        }

    def predict_from_geometry(model, geometry):
        return model["data_processor"].torch_predict(
            model["prediction_function"], geometry
        )

    @api.post("/models/{model_name}/predict", response_class=FileResponse)
    async def predict(
        model_name: ModelName,
        geometry_file: UploadFile,
        background_tasks: BackgroundTasks,
    ):
        """Returns a prediction of the matrix given an uploaded geometry file."""
        # Find out which parser should we use given the file name.
        cls = sisl.get_sile_class(geometry_file.filename)

        # Parse the contents of the file into text, and wrap it in a StringIO object.
        with geometry_file.file as f:
            content = StringIO(f.read().decode("utf-8"))

        # Make sisl read the geometry from the StringIO object.
        with cls(content) as sile:
            geometry = sile.read_geometry()

        # Get model.
        model = models[model_name.value]

        matrix = predict_from_geometry(model, geometry)

        # WRITE THE MATRIX TO A TEMPORARY FILE
        tmp_file = tempfile.NamedTemporaryFile(suffix=".DM", delete=False)
        file_path = Path(tmp_file.name)
        matrix.write(file_path)

        # We need to delete the file after the response is sent, so we add the deletion
        # to the background tasks, which run after sending the response.
        background_tasks.add_task(file_path.unlink, missing_ok=True)

        # Return the file.
        return FileResponse(
            file_path, media_type="application/octet-stream", filename=file_path.name
        )

    if local:

        @api.get("/models/{model_name}/local_write_predict")
        async def local_write_predict(
            model_name: ModelName,
            geometry_path: str,
            output_path: Optional[str] = None,
            allow_overwrite: bool = False,
        ) -> str:
            """Given the path to a geometry file, writes the predicted matrix to a file.

            This is the same as predict, but considering that the server can interact directly
            with the file system that the user is targeting. Therefore, the input geometry is
            passed as a path to a file, and the output is also a path to a file.

            NOTE: If the output is not an absolute path, it will be interpreted as
            relative to the directory of the input file. If not specified, the
            default name for the app will be used.

            Returns
            -------
            str
                Absolute path to the file where the matrix was written.
            """
            model = models[model_name]

            # First, the input file, which we resolve to an absolute path to avoid
            # ambiguity.
            runfile = Path(geometry_path).resolve()

            # Then the output path where we should write the matrix.
            if output_path is None:
                output_path = model.get("output_file_name")

            out_file = Path(output_path).resolve()
            if not out_file.is_absolute():
                out_file = runfile.parent / out_file
            out_file = Path(out_file).resolve()

            geometry = sisl.get_sile(runfile).read_geometry()

            # USE THE MODEL
            matrix = predict_from_geometry(model, geometry)

            if allow_overwrite and out_file.exists():
                raise ValueError(
                    f"Output file {out_file} already exists and overwrite is not allowed."
                )

            # And write the matrix to it.
            matrix.write(out_file)

            return str(out_file)

    else:

        @api.get("/models/{model_name}/local_write_predict")
        async def local_write_predict(
            model_name: ModelName,
            geometry_path: str,
            output_path: Optional[str] = None,
            allow_overwrite: bool = False,
        ) -> str:
            """Given the path to a geometry file, writes the predicted matrix to a file.

            This is the same as predict, but considering that the server can interact directly
            with the file system that the user is targeting. Therefore, the input geometry is
            passed as a path to a file, and the output is also a path to a file.

            NOTE: If the output is not an absolute path, it will be interpreted as
            relative to the directory of the input file. If not specified, the
            default name for the app will be used.

            Returns
            -------
            str
                Absolute path to the file where the matrix was written.
            """
            raise HTTPException(
                status_code=403, detail="This server does not allow local writes."
            )

    @api.post("/models/{model_name}/test")
    async def test(
        model_name: ModelName,
        geometry_file: UploadFile,
        output_matrix_file: UploadFile,
        background_tasks: BackgroundTasks,
    ):
        """Tests how good a prediction is given an uploaded geometry file and an uploaded matrix file."""

        # Get model.
        model = models[model_name.value]

        # Find out which parser should we use given the file name.
        cls = sisl.get_sile_class(geometry_file.filename)

        # Parse the contents of the file into text, and wrap it in a StringIO object.
        with geometry_file.file as f:
            content = StringIO(f.read().decode("utf-8"))

        # Make sisl read the geometry from the StringIO object.
        with cls(content) as sile:
            geometry = sile.read_geometry()

        geometry = model["data_processor"].add_basis_to_geometry(geometry)

        # Parse the contents of the matrix file.
        provided_matrix_file = tempfile.NamedTemporaryFile(
            suffix=output_matrix_file.filename, delete=False
        )
        with output_matrix_file.file as f:
            with open(provided_matrix_file.name, "wb") as f2:
                f2.write(f.read())

        matrix_parser = sisl.get_sile_class(output_matrix_file.filename)
        matrix_sile = matrix_parser(provided_matrix_file.name)

        # Find out which type of matrix should we read and read it.
        read_method = f"read_{model['data_processor'].out_matrix}"
        target_matrix = getattr(matrix_sile, read_method)(geometry=geometry)

        with torch.no_grad():
            # USE THE MODEL
            # First, we need to process the input data, to get inputs as the model expects.
            input_data = BasisMatrixTorchData.new(
                target_matrix, data_processor=model["data_processor"], labels=True
            )

            # Then, we run the model.
            out = model["prediction_function"](input_data)

            # And finally, we convert the output to a matrix.
            metrics = model["data_processor"].compute_metrics(
                out, input_data, config_resolved=False
            )

        return metrics

    # Functionality for time series forecasting
    extrapolate_api = create_extrapolation_app(
        matrix_refs={
            k: partial(predict_from_geometry, model) for k, model in models.items()
        },
        data_processors={k: model["data_processor"] for k, model in models.items()},
    )
    api.mount("/extrapolation", extrapolate_api)

    # From here below, we define the endpoints that handle the frontend. It is a very simple
    # frontend using Jinja2 templates.
    templates = Jinja2Templates(
        directory=Path(__file__).parent / "frontend" / "templates"
    )

    @app.get("/models/{model_name}/{model_action}", response_class=HTMLResponse)
    async def get(request: Request, model_action: ModelAction, model_name: ModelName):
        return templates.TemplateResponse(
            f"{model_action.value}_page.html",
            {
                "request": request,
                "selected_model": model_name.value,
                "model_name": model_name.value,
                "model": models[model_name.value],
                "models": models,
                "model_action": actions[model_action],
                "actions": actions,
                "tabs": tabs,
                "selected_tab": tabs[Tab.models],
            },
        )

    @app.get("/models/{model_name}", response_class=HTMLResponse)
    async def get(request: Request, model_name: ModelName):
        return RedirectResponse(
            f"/models/{model_name.value}/{ModelAction.predict.value}"
        )

    @app.get("/models", response_class=HTMLResponse)
    async def get(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "models": models,
                "model_action": actions[ModelAction.predict],
                "actions": actions,
                "tabs": tabs,
                "selected_tab": tabs[Tab.models],
            },
        )

    @app.get("/about", response_class=HTMLResponse)
    async def get(request: Request):
        return templates.TemplateResponse(
            "about.html",
            {
                "request": request,
                "tabs": tabs,
                "selected_tab": tabs[Tab.about],
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def get(request: Request):
        return RedirectResponse(f"/models")

    static = StaticFiles(directory=Path(__file__).parent / "frontend" / "static")
    app.mount("/static", static, name="static")

    return app


def create_server_app_from_filesystem(
    model_files: Dict[str, str] = {},
    local: bool = False,
    cpu: bool = True,
):
    """Launches a server that serves predictions from trained models stored in checkpoint files.

    This function just builds the dictionary of models from the ckpt files and then calls
    ``server_app``.

    Parameters
    ----------
    ckpt_files : Sequence[str]
        List of checkpoint files to load.
    local : bool, optional
        If True, the server allows the user to ask for changes in the local file system.
    cpu : bool, optional
        Load parameters in the CPU regardless of whether they were in the GPU, by default True.
    """
    # Initialize the dictionary that will hold the models.
    models = {}

    # Loop over the ckpt files and load the models.
    for model_name, ckpt_file in model_files.items():
        ckpt_file = Path(ckpt_file).resolve()

        if ckpt_file.is_dir():
            ckpt_file = Path(ckpt_file) / "spec.yaml"

        # If it is a yaml file
        if ckpt_file.suffix == ".yaml":
            import yaml

            # Read the yaml spec
            with open(ckpt_file, "r") as f:
                spec = yaml.safe_load(f)

            spec["root_dir"] = ckpt_file.parent

        else:
            spec = {"files": {"ckpt": ckpt_file}}

        spec["root_dir"] = Path(spec.get("root_dir", "."))

        if Path(spec["files"]["ckpt"]).is_absolute():
            model_ckpt = spec["files"]["ckpt"]
        else:
            model_ckpt = spec["root_dir"] / spec["files"]["ckpt"]

        model, data_processor = load_from_lit_ckpt(model_ckpt, cpu=cpu)
        model.eval()

        models[model_name] = {
            **spec,
            "prediction_function": model,
            "data_processor": data_processor,
        }

    return create_server_app(models, local=local)
