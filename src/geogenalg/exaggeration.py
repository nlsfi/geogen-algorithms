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
        GeoDataFrame containing the narrow polygon parts. **Each row corresponds
        to a row in the input GeoDataFrame.** If an input polygon does not
        contain any parts narrower than the threshold, the corresponding
        geometry in the result will be empty (a GeometryCollection), but the row
        will still be present.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than
              polygon geometries.

    """
    if not all(input_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])):
        msg = "Extract narrow parts only supports Polygon or MultiPolygon geometries."
        raise GeometryTypeError(msg)

    input_gdf.geometry = input_gdf.geometry.buffer(0)
    wide_parts_gdf = input_gdf.copy()

    # Remove polygon parts narrower than threshold
    wide_parts_gdf.geometry = wide_parts_gdf.buffer(
        -0.5 * threshold, cap_style="flat", join_style="mitre"
    ).buffer(0.5 * threshold, cap_style="flat", join_style="mitre")

    # Extract narrow parts by difference
    narrow_parts_gdf = wide_parts_gdf.copy()
    narrow_parts_gdf.geometry = input_gdf.difference(wide_parts_gdf.union_all())

    # Clean small slivers
    narrow_parts_gdf.geometry = narrow_parts_gdf.buffer(
        -0.1, cap_style="flat", join_style="mitre"
    ).buffer(0.1, cap_style="flat", join_style="mitre")

    return narrow_parts_gdf
