#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd
from shapely.geometry import Point

from geogenalg.core.exceptions import GeometryTypeError


def reduce_nearby_points(
    input_gdf: gpd.GeoDataFrame, reduce_threshold: float
) -> gpd.GeoDataFrame:
    """Reduce the number of points by clustering and replacing them with their centroid.

    Args:
    ----
        input_gdf: Input GeoDataFrame with point geometries
        reduce_threshold: Distance used for buffering and clustering

    Returns:
    -------
        A GeoDataFrame containing centroids of clusters.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than point features

    """
    if not input_gdf.geometry.apply(lambda geom: isinstance(geom, Point)).all():
        msg = "reduce_nearby_points only supports Point geometries."
        raise GeometryTypeError(msg)

    result_gdf = input_gdf.copy()

    result_gdf.geometry = result_gdf.buffer(reduce_threshold)

    # Merge overlapping buffers into clusters
    result_gdf = gpd.GeoDataFrame(
        geometry=[result_gdf.geometry.union_all()], crs=input_gdf.crs
    )

    result_gdf = result_gdf.explode(index_parts=False)

    # Calculate the centroid of each cluster
    result_gdf.geometry = result_gdf.centroid

    return result_gdf
