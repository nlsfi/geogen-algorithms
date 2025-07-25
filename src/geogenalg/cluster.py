#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd
from pandas import concat

from geogenalg.core.exceptions import GeometryTypeError


def reduce_nearby_points(
    input_gdf: gpd.GeoDataFrame,
    reduce_threshold: float,
    unique_key_column: str,
) -> gpd.GeoDataFrame:
    """Reduce the number of points by clustering and replacing them with their centroid.

    Args:
    ----
        input_gdf: Input GeoDataFrame with point geometries. The GeoDataFrame must
              include a column with a unique key.
        reduce_threshold: Distance used for buffering and clustering
        unique_key_column: Name of the column containing unique identifiers

    Returns:
    -------
        A GeoDataFrame containing centroids of clusters.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than point features

    """
    if input_gdf.geometry.type.unique().tolist() != ["Point"]:
        msg = "reduce_nearby_points only supports Point geometries."
        raise GeometryTypeError(msg)

    buffered_gdf = input_gdf.copy()

    buffered_gdf.geometry = buffered_gdf.buffer(reduce_threshold)

    dissolved_gdf: gpd.GeoDataFrame = (
        buffered_gdf.dissolve().explode(index_parts=True).reset_index(drop=True)
    )

    clustered_points_gdfs: list[gpd.GeoDataFrame] = []

    for cluster_polygon in dissolved_gdf.geometry:
        in_cluster_gdf = input_gdf[input_gdf.geometry.within(cluster_polygon)]

        min_id = in_cluster_gdf[unique_key_column].min()
        representative_point_gdf: gpd.GeoDataFrame = in_cluster_gdf.loc[
            in_cluster_gdf[unique_key_column] == min_id
        ].copy()

        representative_point_gdf = representative_point_gdf.set_geometry(
            [cluster_polygon.centroid]
        )

        # If several points are clustered, their keys are saved as cluster members
        if len(in_cluster_gdf) > 1:
            representative_point_gdf["cluster_members"] = [
                in_cluster_gdf[unique_key_column].tolist()
            ]

        # For single points, the cluster members field is set to None
        else:
            representative_point_gdf["cluster_members"] = None

        clustered_points_gdfs.append(representative_point_gdf)

    return concat(clustered_points_gdfs)
