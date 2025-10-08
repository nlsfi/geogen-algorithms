#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Annotated

import typer
from geopandas import read_file

from geogenalg.application.generalize_clusters_to_centroids import (
    GeneralizePointClustersAndPolygonsToCentroids,
)


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

    """
    uri = GeoPackageURI(value, None)

    expected_split_parts = 2

    stripped = value.strip("\"'")

    if "|" in stripped:
        split = stripped.split("|")
        if len(split) != expected_split_parts:
            return uri

        file = split[0]
        layer_name = split[1]
        uri.file = file
        uri.layer_name = layer_name

    return uri


GeoPackageArgument = Annotated[
    GeoPackageURI,
    typer.Argument(
        parser=geopackage_uri,
        help=(
            "Path to a GeoPackage, with layer name optionally specified, "
            + 'examples: "my_geopackage.gpkg" "my_geopackage.gpkg|my_layer_name"'
        ),
    ),
]

GeoPackageOption = Annotated[
    GeoPackageURI | None,
    typer.Option(
        parser=geopackage_uri,
        help=(
            "Path to a GeoPackage, with layer name optionally specified, "
            + 'examples: "my_geopackage.gpkg" "my_geopackage.gpkg|my_layer_name"'
        ),
    ),
]

app = typer.Typer()


@app.command()
def clusters_to_centroids(
    input_geopackage: GeoPackageArgument,
    output_geopackage: GeoPackageArgument,
    unique_id_column: Annotated[str, typer.Option()],
    mask_data: GeoPackageOption = None,
    cluster_distance: float = 30.0,
    polygon_min_area: float = 4000.0,
    feature_type_column: str = "feature_type",
) -> None:
    """Execute Generalize point clusters and polygons to centroids algorithm."""
    algorithm = GeneralizePointClustersAndPolygonsToCentroids(
        cluster_distance=cluster_distance,
        polygon_min_area=polygon_min_area,
        unique_id_column=unique_id_column,
        feature_type_column=feature_type_column,
        aggregation_functions=None,
    )

    if mask_data is not None:
        mask_gdf = read_file(mask_data.file, layer=mask_data.layer_name)
        reference_data = {"mask": mask_gdf}
    else:
        reference_data = {}

    in_gdf = read_file(input_geopackage.file, layer=input_geopackage.layer_name)
    output = algorithm.execute(in_gdf, reference_data=reference_data)
    output.to_file(output_geopackage.file, layer=output_geopackage.layer_name)


def main() -> None:
    """Execute typer application."""
    app()


if __name__ == "__main__":
    main()
