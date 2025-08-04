#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

from geogenalg import cluster, displacement


@dataclass
class AlgorithmOptions:
    """Options for generalize points algorithm.

    Attributes:
        reduce_threshold: Distance used for buffering and clustering points
        displace_threshold: Minimum allowed distance between points after displacement
        displace_points_iterations: The number of times to repeat displacement loop
        unique_key_column: Name of the column containing unique identifiers
        cluster_members_column: Name of the column that lists the points included in
              each cluster

    """

    reduce_threshold: float
    displace_threshold: float
    displace_points_iterations: int
    unique_key_column: str
    cluster_members_column: str


def create_generalized_points(
    input_path: Path | str,
    layer_name: str,
    options: AlgorithmOptions,
    output_path: str,
) -> None:
    """Create GeoDataFrame and pass it to the generalization function.

    Args:
    ----
        input_path: Path to the input GeoPackage
        layer_name: Name of the point layer
        options: Algorithm parameters for generalize points
        output_path: Path to save the output GeoPackage

    Raises:
    ------
        FileNotFoundError: If the input_path does not exist

    """
    if not Path(input_path).resolve().exists():
        raise FileNotFoundError

    points_gdf = gpd.read_file(input_path, layer=layer_name)

    result_gdf = generalize_points(points_gdf, options)

    result_gdf.to_file(output_path, driver="GPKG")


def generalize_points(
    points_gdf: gpd.GeoDataFrame,
    options: AlgorithmOptions,
) -> gpd.GeoDataFrame:
    """Generalize point features by reducing and displacing them.

    Note:
        If more than two points are located close to each other, the exact
        displace_threshold may not be achieved. Increasing the number of iterations for
        displacing points will bring the minimum distances closer to the threshold.

    Args:
    ----
        points_gdf: A GeoDataFrame containing the point features
        options: Algorithm parameters for generalize points

    Returns:
    -------
        A GeoDataFrame containing generalized points

    """
    result_gdf = points_gdf.copy()

    result_gdf = cluster.reduce_nearby_points(
        result_gdf,
        options.reduce_threshold,
        options.unique_key_column,
        options.cluster_members_column,
    )

    return displacement.displace_points(
        result_gdf, options.displace_threshold, options.displace_points_iterations
    )
