#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import geopandas as gpd

from geogenalg.core.exceptions import GeometryTypeError


def extract_narrow_polygon_parts(
    input_gdf: gpd.GeoDataFrame, threshold: float
) -> gpd.GeoDataFrame:
    """Extract polygon parts narrower than the threshold.

    Args:
    ----
        input_gdf: GeoDataFrame containing Polygon or MultiPolygon geometries
        threshold: Minimum width for a polygon part. Parts narrower than this are
              extracted.

    Returns:
    -------
        GeoDataFrame containing only the narrow polygon parts.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than
              polygon geometries.

    """
    if not all(input_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])):
        msg = "Separate narrow parts only supports Polygon or MultiPolygon geometries."
        raise GeometryTypeError(msg)

    result_gdf = input_gdf.copy()

    # Remove polygon parts narrower than threshold
    result_gdf.geometry = result_gdf.buffer(
        (-0.5 * threshold), cap_style="flat", join_style="mitre"
    )
    result_gdf.geometry = result_gdf.buffer(
        (0.5 * threshold), cap_style="flat", join_style="mitre"
    )

    # Extract narrow parts by difference
    result_gdf.geometry = input_gdf.difference(result_gdf.union_all())

    # Clean small slivers
    result_gdf.geometry = result_gdf.buffer(
        -0.1, cap_style="flat", join_style="mitre"
    ).buffer(0.1, cap_style="flat", join_style="mitre")

    return result_gdf
