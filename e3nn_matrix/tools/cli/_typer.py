import typing
from typing_extensions import Annotated

from enum import Enum

import inspect
from copy import copy
import yaml

import typer


# Classes that hold information regarding how a given parameter should behave in a CLI
# They are meant to be used as metadata for the type annotations. That is, passing them
# to Annotated. E.g.: Annotated[int, CLIArgument(option="some_option")]. Even if they
# are empty, they indicate whether to treat the parameter as an argument or an option.
class CLIArgument:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class CLIOption:
    def __init__(self, *param_decls: str, **kwargs):
        if len(param_decls) > 0:
            kwargs["param_decls"] = param_decls
        self.kwargs = kwargs


def get_params_help(func) -> dict:
    """Gets the text help of parameters from the docstring"""
    params_help = {}

    in_parameters = False
    read_key = None
    arg_content = ""

    for line in func.__doc__.split("\n"):
        if "Parameters" in line:
            in_parameters = True
            space = line.find("Parameters")
            continue

        if in_parameters:
            if len(line) < space + 1:
                continue
            if len(line) > 1 and line[0] != " ":
                break

            if line[space] not in (" ", "-"):
                if read_key is not None:
                    params_help[read_key] = arg_content

                read_key = line.split(":")[0].strip()
                arg_content = ""
            else:
                if arg_content == "":
                    arg_content = line.strip()
                    arg_content = arg_content[0].upper() + arg_content[1:]
                else:
                    arg_content += " " + line.strip()

        if line.startswith("------"):
            break

    if read_key is not None:
        params_help[read_key] = arg_content

    return params_help


def get_dict_param_kwargs(dict_annotation_args):
    def yaml_dict(d: str):
        if isinstance(d, dict):
            return d

        return yaml.safe_load(d)

    argument_kwargs = {"parser": yaml_dict}

    if len(dict_annotation_args) == 2:
        try:
            argument_kwargs[
                "metavar"
            ] = f"YAML_DICT[{dict_annotation_args[0].__name__}: {dict_annotation_args[1].__name__}]"
        except:
            argument_kwargs[
                "metavar"
            ] = f"YAML_DICT[{dict_annotation_args[0]}: {dict_annotation_args[1]}]"

    return argument_kwargs


# This dictionary keeps the kwargs that should be passed to typer arguments/options
# for a given type. This is for example to be used for types that typer does not
# have built in support for.
_CUSTOM_TYPE_KWARGS = {
    dict: get_dict_param_kwargs,
}


def _get_custom_type_kwargs(type_):
    if hasattr(type_, "__metadata__"):
        type_ = type_.__origin__

    if typing.get_origin(type_) is not None:
        args = typing.get_args(type_)
        type_ = typing.get_origin(type_)
    else:
        args = ()

    try:
        argument_kwargs = _CUSTOM_TYPE_KWARGS.get(type_, {})
        if callable(argument_kwargs):
            argument_kwargs = argument_kwargs(args)
    except:
        argument_kwargs = {}

    return argument_kwargs


def annotate_typer(func):
    """Annotates a function for a typer app.

    It returns a new function, the original function is not modified.
    """
    # Get the help message for all parameters found at the docstring
    params_help = get_params_help(func)

    # Get the original signature of the function
    sig = inspect.signature(func)

    # Loop over parameters in the signature, modifying them to include the
    # typer info.
    new_parameters = []
    for param in sig.parameters.values():
        argument_kwargs = _get_custom_type_kwargs(param.annotation)

        default = param.default
        if isinstance(param.default, Enum):
            default = default.value

        typer_arg_cls = (
            typer.Argument if param.default == inspect.Parameter.empty else typer.Option
        )
        if hasattr(param.annotation, "__metadata__"):
            for meta in param.annotation.__metadata__:
                if isinstance(meta, CLIArgument):
                    typer_arg_cls = typer.Argument
                    argument_kwargs.update(meta.kwargs)
                elif isinstance(meta, CLIOption):
                    typer_arg_cls = typer.Option
                    argument_kwargs.update(meta.kwargs)

        if "param_decls" in argument_kwargs:
            argument_args = argument_kwargs.pop("param_decls")
        else:
            argument_args = []

        new_parameters.append(
            param.replace(
                default=default,
                annotation=Annotated[
                    param.annotation,
                    typer_arg_cls(
                        *argument_args,
                        help=params_help.get(param.name),
                        **argument_kwargs,
                    ),
                ],
            )
        )

    # Create a copy of the function and update it with the modified signature.
    # Also remove parameters documentation from the docstring.
    annotated_func = copy(func)

    annotated_func.__signature__ = sig.replace(parameters=new_parameters)
    annotated_func.__doc__ = func.__doc__[: func.__doc__.find("Parameters\n")]

    return annotated_func


# ----------------------------------------------------
#           Typer markdown patch
# ----------------------------------------------------
# This is a patch for typer to allow for markdown in the help messages (see https://github.com/tiangolo/typer/issues/678)

import inspect
from typing import Union, Iterable

import click
from rich.console import group
from rich.markdown import Markdown
from rich.text import Text
from typer.core import MarkupMode
from typer.rich_utils import (
    MARKUP_MODE_MARKDOWN,
    STYLE_HELPTEXT_FIRST_LINE,
    _make_rich_rext,
)


@group()
def _get_custom_help_text(
    *,
    obj: Union[click.Command, click.Group],
    markup_mode: MarkupMode,
) -> Iterable[Union[Markdown, Text]]:
    # Fetch and dedent the help text
    help_text = inspect.cleandoc(obj.help or "")

    # Trim off anything that comes after \f on its own line
    help_text = help_text.partition("\f")[0]

    # Get the first paragraph
    first_line = help_text.split("\n\n")[0]
    # Remove single linebreaks
    if markup_mode != MARKUP_MODE_MARKDOWN and not first_line.startswith("\b"):
        first_line = first_line.replace("\n", " ")
    yield _make_rich_rext(
        text=first_line.strip(),
        style=STYLE_HELPTEXT_FIRST_LINE,
        markup_mode=markup_mode,
    )

    # Get remaining lines, remove single line breaks and format as dim
    remaining_paragraphs = help_text.split("\n\n")[1:]
    if remaining_paragraphs:
        remaining_lines = inspect.cleandoc(
            "\n\n".join(remaining_paragraphs).replace("<br/>", "\\")
        )
        yield _make_rich_rext(
            text=remaining_lines,
            style="cyan",
            markup_mode=markup_mode,
        )


import typer.rich_utils

typer.rich_utils._get_help_text = _get_custom_help_text
