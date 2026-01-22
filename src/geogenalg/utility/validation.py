#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import Literal

from geopandas import GeoDataFrame, GeoSeries

ShapelyGeometryTypeString = Literal[
    "Point",
    "LineString",
    "LinearRing",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
]


def check_gdf_geometry_type(
    gdf: GeoDataFrame,
    accepted_types: set[ShapelyGeometryTypeString],
) -> bool:
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
    geoseries: GeoSeries,
    accepted_types: set[ShapelyGeometryTypeString],
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
