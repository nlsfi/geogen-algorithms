#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd


def drop_empty_geometries(input_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame without empty or missing geometries.

    Args:
    ----
        input_gdf: The input GeoDataFrame

    Returns:
    -------
        GeoDataFrame containing only valid, non-empty geometries.

    """
    return input_gdf[~(input_gdf.geometry.is_empty | input_gdf.geometry.isna())]


def fix_invalid_geometries(input_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Repair invalid geometries in a GeoDataFrame using ``make_valid()``.

    Args:
    ----
        input_gdf: The input GeoDataFrame containing potentially invalid geometries

    Returns:
    -------
        GeoDataFrame where invalid geometries have been repaired.

    """
    result_gdf = input_gdf.copy()
    result_gdf.geometry = result_gdf.make_valid()
    return result_gdf
