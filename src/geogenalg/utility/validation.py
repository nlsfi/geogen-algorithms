#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from geopandas import GeoDataFrame, GeoSeries


def check_gdf_geometry_type(gdf: GeoDataFrame, accepted_types: list[str]) -> bool:
    """Check if all geometries are of acceptable type.

    Args:
    ----
        gdf: GeoDataFrame to be checked.
        accepted_types: List of accepted geometry types as strings (in the same
            format as GeoSeries.geom_type).

    Returns:
    -------
        True if all geometries are of accepted type, False if not.

    """
    return all(gdf.geometry.type.isin(accepted_types))


def check_geoseries_geometry_type(
    geoseries: GeoSeries, accepted_types: list[str]
) -> bool:
    """Check if all geometries are of acceptable type.

    Args:
    ----
        geoseries: GeoSeries to be checked.
        accepted_types: List of accepted geometry types as strings (in the same
            format as GeoSeries.geom_type).

    Returns:
    -------
        True if all geometries are of accepted type, False if not.

    """
    return all(geoseries.type.isin(accepted_types))
