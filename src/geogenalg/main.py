#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from ast import AnnAssign, Assign, ClassDef, Constant, Expr, Name, parse
from dataclasses import dataclass
from inspect import Parameter, cleandoc, getfullargspec, getsource, signature
from itertools import pairwise
from textwrap import dedent
from types import FunctionType
from typing import Annotated, Any, cast

import typer
from geopandas import GeoDataFrame, read_file
from pandas import concat

from geogenalg.application import BaseAlgorithm
from geogenalg.application.generalize_clusters_to_centroids import (
    GeneralizePointClustersAndPolygonsToCentroids,
)
from geogenalg.application.generalize_fences import GeneralizeFences
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

GEOPACKAGE_URI_HELP = (
    "Path to a GeoPackage, with layer name optionally specified, "
    + 'examples: "my_geopackage.gpkg" "my_geopackage.gpkg|my_layer_name"'
)


@dataclass(frozen=True)
class GeoPackageURI:
    """Custom input type for CLI.

    Allows specifying GeoPackage file and picking a specific layer in one argument.
    """

    file: str
    layer_name: str | None


@dataclass(frozen=True)
class NamedGeoPackageURI:
    """Custom input type for CLI.

    Allows specifying a GeoPackageURI by name, to be used as reference data in
    algorithms.
    """

    name: str
    uri: GeoPackageURI


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

    accepted_file_layer_delimiters = ("|", "@")

    if all(delimiter in stripped for delimiter in accepted_file_layer_delimiters):
        msg = "Incorrectly formatted GeoPackageURI"
        raise typer.BadParameter(msg)

    for delimiter in accepted_file_layer_delimiters:
        if delimiter in stripped:
            split = stripped.split(delimiter)
            if len(split) != expected_split_parts:
                msg = "Incorrectly formatted GeoPackageURI"
                raise typer.BadParameter(msg)

            file = split[0]
            layer_name = split[1]

            return GeoPackageURI(file=file, layer_name=layer_name)

    return uri


def named_geopackage_uri(value: str) -> NamedGeoPackageURI:
    """Parse string into NamedGeoPackageURI.

    Args:
    ----
        value: string to parse

    Returns:
    -------
        Parsed NamedGeoPackageURI

    Raises:
    ------
        BadParameter: if string is incorrectly formatted

    """
    name_uri_delimiter = ":"
    expected_split_parts = 2

    split = value.split(name_uri_delimiter)

    name = split[0]

    if len(split) != expected_split_parts or not name:
        msg = "Incorrectly formatted NamedGeoPackageURI"
        raise typer.BadParameter(msg)

    uri = geopackage_uri(split[1])

    return NamedGeoPackageURI(name=name, uri=uri)


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
        help=GEOPACKAGE_URI_HELP,
    ),
]

GeoPackageOptionMandatory = Annotated[
    GeoPackageURI,
    typer.Option(
        parser=geopackage_uri,
        help=(
            "Path to a GeoPackage, with layer name optionally specified, "
            + 'examples: "my_geopackage.gpkg" "my_geopackage.gpkg|my_layer_name"'
        ),
    ),
]

ReferenceGeoPackageList = Annotated[
    list[NamedGeoPackageURI],
    typer.Option(
        parser=named_geopackage_uri,
        help=(
            'Reference data as name and GeoPackageURI. Examples: "--ref=name:data.gpkg"'
            + '--ref=name:data.gpkg@layer". You can specify the option multiple times. '
            + "These will be used as reference data in the algorithm. The name should "
            + "correspond to a reference key. If multiple datasets with the same name "
            + "are specified, they will be combined."
        ),
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

        reference_options = cast("list[NamedGeoPackageURI]", args.pop("ref"))

        reference_data: dict[str, GeoDataFrame] = {}
        for reference in reference_options:
            reference_gdf = read_file(
                reference.uri.file,
                layer=reference.uri.layer_name,
            )

            if reference.name not in reference_data:
                reference_data[reference.name] = reference_gdf
            else:
                reference_data[reference.name] = concat(
                    [
                        reference_data[reference.name],
                        reference_gdf,
                    ]
                )

        instance = algorithm(**kwargs)

        if unique_id_column is not None:
            in_gdf = read_gdf_from_file_and_set_index(
                input_geopackage.file,
                unique_id_column,
                layer=input_geopackage.layer_name,
            )
        else:
            in_gdf = read_file(input_geopackage.file, layer=input_geopackage.layer_name)

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
    }

    for cli_command_name, alg in commands_and_algs.items():
        algorithm_command_function = _function_generator(alg)
        algorithm_command_function.__name__ = cli_command_name

        function_signature = signature(algorithm_command_function)

        unique_id_column = os.environ.get("GEOGENALG_UNIQUE_ID_COLUMN")

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
            Parameter(
                name="unique_id_column",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=unique_id_column,
                annotation=Annotated[
                    str | None,
                    typer.Option(
                        help=(
                            "Column containing unique id for features. If specified, "
                            + "the column will be used as an index. May also be set as "
                            + "environment variable GEOGENALG_UNIQUE_ID_COLUMN."
                        ),
                    ),
                ],
            ),
            Parameter(
                name="ref",
                default=[],
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=ReferenceGeoPackageList,
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


@app.command()
def fences(
    input_geopackage: GeoPackageArgument,
    output_geopackage: GeoPackageArgument,
    masts_data: GeoPackageOptionMandatory,
    closing_fence_area_threshold: Annotated[float, typer.Option()],
    closing_fence_area_with_mast_threshold: Annotated[float, typer.Option()],
    fence_length_threshold: Annotated[float, typer.Option()],
    fence_length_threshold_in_closed_area: Annotated[float, typer.Option()],
    simplification_tolerance: Annotated[float, typer.Option()],
    gap_threshold: Annotated[float, typer.Option()],
    attribute_for_line_merge: Annotated[str, typer.Option()],
) -> None:
    """Execute Generalize point clusters and polygons to centroids algorithm."""
    algorithm = GeneralizeFences(
        closing_fence_area_threshold=closing_fence_area_threshold,
        closing_fence_area_with_mast_threshold=closing_fence_area_with_mast_threshold,
        fence_length_threshold=fence_length_threshold,
        fence_length_threshold_in_closed_area=fence_length_threshold_in_closed_area,
        simplification_tolerance=simplification_tolerance,
        gap_threshold=gap_threshold,
        attribute_for_line_merge=attribute_for_line_merge,
    )

    masts_gdf = read_file(masts_data.file, layer=masts_data.layer_name)
    reference_data = {"masts": masts_gdf}

    in_gdf = read_file(input_geopackage.file, layer=input_geopackage.layer_name)
    output = algorithm.execute(in_gdf, reference_data=reference_data)
    output.to_file(output_geopackage.file, layer=output_geopackage.layer_name)


def main() -> None:
    """Build commands for each algorithm and execute typer CLI app."""
    build_app()

    app()


if __name__ == "__main__":
    main()
