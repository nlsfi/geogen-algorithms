#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
import os
from inspect import Parameter, signature
from types import FunctionType
from typing import Annotated, Any, cast

from geogenalg.application.generalize_address_points import GeneralizeAddressPoints
from geogenalg.application.generalize_conservation_areas import (
    GeneralizeConservationAreas,
)
from geogenalg.application.generalize_contours import GeneralizeContours
from geogenalg.application.generalize_railroads import GeneralizeRailroads
from geogenalg.application.generalize_slopelines import GeneralizeSlopeLines
from geogenalg.application.generalize_watercourse_lines import (
    GeneralizeWaterCourseLines,
)
from geogenalg.utility.cli import (
    IGNORED_TYPES_FOR_CLI,
    GeoPackageURI,
    NamedGeoPackageURI,
    TransformedTypeInformation,
    basemetadata_to_string,
    get_basealgorithm_attribute_docstrings,
    int_or_str_list,
)

try:
    import typer
except ImportError:
    print("CLI dependency not found. Install 'geogenalg[cli]' to use this command.")  # noqa: T201
    print("Or during development install by running 'uv sync --extra=cli'.")  # noqa: T201
    import sys

    sys.exit(0)

from geopandas import GeoDataFrame, read_file

from geogenalg.application import BaseAlgorithm
from geogenalg.application.dissolve_polygons import DissolvePolygons
from geogenalg.application.generalize_building_areas import GeneralizeBuildingAreas
from geogenalg.application.generalize_buildings import GeneralizeBuildings
from geogenalg.application.generalize_cliffs import GeneralizeCliffs
from geogenalg.application.generalize_fences import GeneralizeFences
from geogenalg.application.generalize_landcover import GeneralizeLandcover
from geogenalg.application.generalize_points import GeneralizePoints
from geogenalg.application.generalize_polygons_to_points import (
    GeneralizePolygonsToPoints,
)
from geogenalg.application.generalize_power_lines import GeneralizePowerLines
from geogenalg.application.generalize_roads import GeneralizeRoads
from geogenalg.application.generalize_shared_paths import GeneralizeSharedPaths
from geogenalg.application.generalize_water_areas import GeneralizeWaterAreas
from geogenalg.application.generalize_watercourse_areas import (
    GeneralizeWaterCourseAreas,
)
from geogenalg.application.keep_intersection import KeepIntersection
from geogenalg.application.remove_overlap import RemoveOverlap
from geogenalg.utility.dataframe_processing import (
    combine_gdfs,
    read_gdf_from_file_and_set_index,
)

GEOPACKAGE_URI_HELP = (
    "Path to a GeoPackage, with layer name optionally specified after a @ or | symbol, "
    'examples: "my_geopackage.gpkg" "my_geopackage.gpkg@my_layer_name" '
    '"file.gpkg|layer"'
)


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


def input_geopackage_uris(value: str) -> list[GeoPackageURI]:
    """Parse string into a list of GeoPackageURIs.

    Args:
    ----
        value: string to parse

    Returns:
    -------
        List of GeoPackageURIs.

    """
    return [geopackage_uri(uri) for uri in value.split("+")]


MultipleGeoPackagesArgument = Annotated[
    GeoPackageURI,
    typer.Argument(
        parser=input_geopackage_uris,
        help=GEOPACKAGE_URI_HELP
        + ". You can also specify multiple inputs by delimiting them with the '+' "
        + 'symbol ("my_geopackage.gpkg@layer+other_geopackage.gpkg@other_layer"). '
        + "In this case the given inputs will be combined. This can be "
        + "used to f.e. combine datasets with different geometry types to use "
        + "with algorithms which accept multiple geometry types.",
    ),
]

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

ReferenceGeoPackageList = Annotated[
    list[NamedGeoPackageURI],
    typer.Option(
        parser=named_geopackage_uri,
        help=(
            'Reference data as name and GeoPackageURI. Examples: "--ref=name:data.gpkg"'
            '--ref=name:data.gpkg@layer". You can specify the option multiple times. '
            "These will be used as reference data in the algorithm. The name should "
            "correspond to a reference key. If multiple datasets with the same name "
            "are specified, they will be combined."
        ),
    ),
]


app = typer.Typer()


def _function_generator(algorithm: type[BaseAlgorithm]) -> FunctionType:
    def _command_function(
        **kwargs: object,
    ) -> None:
        args = kwargs

        input_geopackages = cast(
            "list[GeoPackageURI]",
            args.pop("input_geopackages"),
        )
        output_geopackage = cast("GeoPackageArgument", args.pop("output_geopackage"))
        unique_id_column = cast("str | None", args.pop("unique_id_column"))

        reference_options = cast("list[NamedGeoPackageURI]", args.pop("ref"))

        reference_data: dict[str, GeoDataFrame] = {}
        for reference in reference_options:
            if unique_id_column is not None:
                reference_gdf = read_gdf_from_file_and_set_index(
                    reference.uri.file,
                    unique_id_column,
                    layer=reference.uri.layer_name,
                )
            else:
                reference_gdf = read_file(
                    reference.uri.file,
                    layer=reference.uri.layer_name,
                )

            if reference.name not in reference_data:
                reference_data[reference.name] = reference_gdf
            else:
                reference_data[reference.name] = combine_gdfs(
                    [
                        reference_data[reference.name],
                        reference_gdf,
                    ]
                )

        # Transform lists to frozensets
        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = frozenset(value)

        instance = algorithm(**kwargs)  # ty: ignore[invalid-argument-type]

        input_gdfs = []
        for uri in input_geopackages:
            if unique_id_column is not None:
                in_gdf = read_gdf_from_file_and_set_index(
                    uri.file,
                    unique_id_column,
                    layer=uri.layer_name,
                )
            else:
                in_gdf = read_file(uri.file, layer=uri.layer_name)

            input_gdfs.append(in_gdf)

        in_gdf = input_gdfs[0] if len(input_gdfs) == 1 else combine_gdfs(input_gdfs)

        output = instance.execute(in_gdf, reference_data=reference_data)
        output.to_file(output_geopackage.file, layer=output_geopackage.layer_name)

    return _command_function


def build_app() -> None:
    """Add commands to typer app from algorithms.

    Exists as a separate function mainly to enable running CLI test.

    """
    commands_and_algs = {
        "address_points": GeneralizeAddressPoints,
        "buildings": GeneralizeBuildings,
        "building_areas": GeneralizeBuildingAreas,
        "cliffs": GeneralizeCliffs,
        "conservation_areas": GeneralizeConservationAreas,
        "contours": GeneralizeContours,
        "dissolve_polygons": DissolvePolygons,
        "fences": GeneralizeFences,
        "keep_intersection": KeepIntersection,
        "landcover": GeneralizeLandcover,
        "points": GeneralizePoints,
        "polygons_to_points": GeneralizePolygonsToPoints,
        "power_lines": GeneralizePowerLines,
        "railroads": GeneralizeRailroads,
        "remove_overlap": RemoveOverlap,
        "roads": GeneralizeRoads,
        "shared_paths": GeneralizeSharedPaths,
        "slope_lines": GeneralizeSlopeLines,
        "watercourse_areas": GeneralizeWaterCourseAreas,
        "watercourse_lines": GeneralizeWaterCourseLines,
        "water_areas": GeneralizeWaterAreas,
    }

    for cli_command_name, alg in commands_and_algs.items():
        algorithm_command_function = _function_generator(alg)
        algorithm_command_function.__name__ = cli_command_name

        function_signature = signature(algorithm_command_function)

        unique_id_column = os.environ.get("GEOGENALG_UNIQUE_ID_COLUMN")

        parameters = [
            Parameter(
                name="input_geopackages",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=MultipleGeoPackagesArgument,
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
                            "the column will be used as an index. May also be set as "
                            "environment variable GEOGENALG_UNIQUE_ID_COLUMN."
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

        # Transform certain types defined on the algorithm-level to a helper
        # type with a custom parser.
        transformed_types_for_cli: dict[type, TransformedTypeInformation] = {
            # Typer doesn't handle sets so read them as lists, all of which are
            # transformed to frozensets in _function_generator()
            frozenset[int | str]: TransformedTypeInformation(
                transformed_type=list[Any],
                parser=int_or_str_list,
                custom_default=frozenset(),
                extra_parse_help=(
                    "Either an integer or a string. Can be specified multiple times. "
                    "If you need to pass an integer as a string, you can add do so "
                    "by appending str: in front of the integer, i.e. str:10."
                ),
            ),
            frozenset[str]: TransformedTypeInformation(
                transformed_type=list[str],
                custom_default=frozenset(),
                extra_parse_help="Can be specified multiple times.",
            ),
            frozenset[int]: TransformedTypeInformation(
                transformed_type=list[int],
                extra_parse_help="Can be specified multiple times.",
            ),
        }

        docstrings = get_basealgorithm_attribute_docstrings(alg)

        for field_name, field_info in alg.model_fields.items():
            type_annotation = field_info.annotation

            if type_annotation in IGNORED_TYPES_FOR_CLI:
                continue

            docstring = docstrings[field_name]

            default = field_info.default

            if isinstance(default, int | float) and field_info.metadata:
                docstring += " Must be "
                limits = " and ".join(
                    basemetadata_to_string(metadata) for metadata in field_info.metadata
                )
                docstring += limits + "."

            if type_annotation in transformed_types_for_cli:
                transform_information = transformed_types_for_cli[type_annotation]

                help_str = (
                    docstring
                    if not transform_information.extra_parse_help
                    else docstring + f" {transform_information.extra_parse_help}"
                )

                type_annotation = Annotated[
                    transform_information.transformed_type,  # ty: ignore[invalid-type-form]
                    typer.Option(
                        parser=transform_information.parser,
                        help=help_str,
                    ),
                ]

                if transform_information.custom_default is not None:
                    default = transform_information.custom_default
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

        # ty complains about that the __signature__ attribute is missing
        # however, setting is required to get the correct parameters showing in
        # the CLI and passing into the actual command function
        algorithm_command_function.__signature__ = function_signature.replace(  # ty: ignore[unresolved-attribute]
            parameters=parameters
        )

        app.pretty_exceptions_show_locals = False
        app.command(help=alg.__doc__)(algorithm_command_function)


def main() -> None:
    """Build commands for each algorithm and execute typer CLI app."""
    build_app()

    app()


if __name__ == "__main__":
    main()
