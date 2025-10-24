#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from ast import AnnAssign, Assign, ClassDef, Constant, Expr, Name, parse
from inspect import Parameter, cleandoc, getfullargspec, getsource, signature
from itertools import pairwise
from textwrap import dedent
from types import FunctionType
from typing import Annotated, Any, cast

import typer
from geopandas import read_file

from geogenalg.application import BaseAlgorithm
from geogenalg.application.generalize_clusters_to_centroids import (
    GeneralizePointClustersAndPolygonsToCentroids,
)
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

GEOPACKAGE_URI_HELP = (
    "Path to a GeoPackage, with layer name optionally specified, "
    + 'examples: "my_geopackage.gpkg" "my_geopackage.gpkg|my_layer_name"'
)
REFERENCE_KEY_ATTRIBUTE_NAME = "reference_key"


class GeoPackageURI:
    """Custom input type for CLI.

    Allows specifying GeoPackage file and picking a specific layer in one argument.
    """

    file: str
    layer_name: str | None

    def __init__(self, file: str, layer_name: str | None) -> None:
        """Set attributes."""
        self.file = file
        self.layer_name = layer_name


def geopackage_uri(value: str) -> GeoPackageURI:
    """Parse string into GeoPackageURI.

    Args:
    ----
        value: string to parse

    Returns:
    -------
        Parsed GeoPackage URI. If layer name could not be parsed, it will be
        None.

    Raises:
    ------
        BadParameter: if string is incorrectly formatted

    """
    uri = GeoPackageURI(value, None)

    expected_split_parts = 2

    stripped = value.strip("\"'")

    if "|" in stripped:
        split = stripped.split("|")
        if len(split) != expected_split_parts:
            msg = "Incorrectly formatted GeoPackageURI"
            raise typer.BadParameter(msg)

        file = split[0]
        layer_name = split[1]
        uri.file = file
        uri.layer_name = layer_name

    return uri


def get_class_attribute_docstrings(cls: type[Any]) -> dict[str, str]:
    """Get any docstrings placed after attribute assignments in a class body.

    Adapted from https://davidism.com/attribute-docstrings/ on 2025-10-24 (MIT
    License).

    Returns
    -------
        Dictionary containing docstring for each attribute found in class.

    Raises
    ------
        TypeError: If input object is not a class.

    """
    cls_node = parse(dedent(getsource(cls))).body[0]

    if not isinstance(cls_node, ClassDef):
        msg = "Given object was not a class."
        raise TypeError(msg)

    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (
            not isinstance(a, Assign | AnnAssign)
            or not isinstance(b, Expr)
            or not isinstance(b.value, Constant)
            or not isinstance(b.value.value, str)
        ):
            continue

        doc = cleandoc(b.value.value)

        targets = a.targets if isinstance(a, Assign) else [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, Name):
                continue

            out[target.id] = doc

    return out


GeoPackageArgument = Annotated[
    GeoPackageURI,
    typer.Argument(
        parser=geopackage_uri,
        help=(GEOPACKAGE_URI_HELP),
    ),
]

GeoPackageOption = Annotated[
    GeoPackageURI | None,
    typer.Option(
        parser=geopackage_uri,
        help=(GEOPACKAGE_URI_HELP),
    ),
]


def _function_generator(algorithm: type[BaseAlgorithm]) -> FunctionType:
    def _command_function(
        **kwargs: GeoPackageArgument | GeoPackageOption | str | float,
    ) -> None:
        args = kwargs

        input_geopackage = cast("GeoPackageArgument", args.pop("input_geopackage"))
        output_geopackage = cast("GeoPackageArgument", args.pop("output_geopackage"))
        unique_id_column = cast("str", args.pop("unique_id_column"))

        try:
            reference_data_key = algorithm.reference_key
        except AttributeError:
            reference_data_key = None
        except:
            raise

        if reference_data_key is not None:
            reference_data_option = cast(
                "GeoPackageOption", args.pop(reference_data_key)
            )

            if reference_data_option is not None:
                reference_gdf = read_file(
                    reference_data_option.file, layer=reference_data_option.layer_name
                )
                reference_data = {reference_data_key: reference_gdf}
            else:
                reference_data = {}
        else:
            reference_data = {}

        instance = algorithm(**kwargs)

        in_gdf = read_gdf_from_file_and_set_index(
            input_geopackage.file,
            unique_id_column,
            layer=input_geopackage.layer_name,
        )

        output = instance.execute(in_gdf, reference_data=reference_data)
        output.to_file(output_geopackage.file, layer=output_geopackage.layer_name)

    return _command_function  # type: ignore[return-value]


app = typer.Typer()


def build_app() -> None:
    """Add commands to typer app from algorithms.

    Exists as a separate function mainly to enable running CLI test.

    Raises
    ------
        ValueError: If reading algorithm attributes fails.

    """
    commands_and_algs = {
        "clusters_to_centroids": GeneralizePointClustersAndPolygonsToCentroids,
        "something_else": GeneralizePointClustersAndPolygonsToCentroids,
    }

    for cli_command_name, alg in commands_and_algs.items():
        algorithm_command_function = _function_generator(alg)
        algorithm_command_function.__name__ = cli_command_name

        function_signature = signature(algorithm_command_function)

        parameters = [
            Parameter(
                name="input_geopackage",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=GeoPackageArgument,
            ),
            Parameter(
                name="output_geopackage",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=GeoPackageArgument,
            ),
            # TODO: make unique_id_column optional?
            Parameter(
                name="unique_id_column",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Annotated[str, typer.Option()],
            ),
        ]

        # TODO: add types into this as needed
        supported_type_hints = (
            float,
            int,
            str,
        )
        argspec = getfullargspec(alg)

        fields = argspec.args

        if len(fields) < 1 and fields[0] != "self":
            msg = "improper arguments found in class"
            raise ValueError(msg)

        fields = fields[1:]

        docstrings = get_class_attribute_docstrings(alg)

        for i, field in enumerate(fields):
            type_annotation = argspec.annotations.get(field)

            if type_annotation not in supported_type_hints:
                continue

            docstring = docstrings[field]

            if argspec.defaults is None:
                msg = "no default values found"
                raise ValueError(msg)

            default = argspec.defaults[i]

            field_name = field

            if field == "reference_key":
                field_name = default
                type_annotation = GeoPackageOption
                default = None
            else:
                type_annotation = Annotated[
                    type_annotation,
                    typer.Option(help=docstring),
                ]

            parameters.append(
                Parameter(
                    name=field_name,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=type_annotation,
                )
            )

        # mypy complains about that the __signature__ attribute is missing
        # however, setting is required to get the correct parameters showing in
        # the CLI and passing into the actual command function
        algorithm_command_function.__signature__ = function_signature.replace(  # type: ignore[attr-defined]
            parameters=parameters
        )

        app.command(help=alg.__doc__)(algorithm_command_function)


def main() -> None:
    """Build commands for each algorithm and execute typer CLI app."""
    build_app()

    app()


if __name__ == "__main__":
    main()
