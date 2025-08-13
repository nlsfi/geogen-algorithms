#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import geopandas as gpd
import pandas as pd


def group_gdfs_by_geometry_type(
    input_gdfs: list[gpd.GeoDataFrame],
) -> dict[str, gpd.GeoDataFrame]:
    """Group and merge GeoDataFeames by their geometry types.

    Args:
    ----
        input_gdfs: A list of GeoDataFrames to be divided

    Returns:
    -------
        A dictionary with keys:
            - 'polygon_gdf': GeoDataFrame of all Polygon and MultiPolygon geometries
            - 'line_gdf': GeoDataFrame of all LineString and MultiLineString geometries
            - 'point_gdf': GeoDataFrame of all Point and MultiPoint geometries
        If no features of a certain type exist, the corresponding GeoDataFrame is empty.

    """
    polygon_gdfs = []
    line_gdfs = []
    point_gdfs = []

    for input_gdf in input_gdfs:
        geom_types = input_gdf.geometry.geom_type.unique()
        if all(geom_type in {"Polygon", "MultiPolygon"} for geom_type in geom_types):
            polygon_gdfs.append(input_gdf)
        elif all(
            geom_type in {"LineString", "MultiLineString"} for geom_type in geom_types
        ):
            line_gdfs.append(input_gdf)
        elif all(geom_type in {"Point", "MultiPoint"} for geom_type in geom_types):
            point_gdfs.append(input_gdf)

    # Get reference CRS from any non-empty GDF
    crs = next((gdf.crs for gdf in input_gdfs if not gdf.empty), None)

    polygon_gdf = (
        gpd.GeoDataFrame(pd.concat(polygon_gdfs, ignore_index=True), crs=crs)
        if polygon_gdfs
        else gpd.GeoDataFrame(geometry=[], crs=crs)
    )
    line_gdf = (
        gpd.GeoDataFrame(pd.concat(line_gdfs, ignore_index=True), crs=crs)
        if line_gdfs
        else gpd.GeoDataFrame(geometry=[], crs=crs)
    )
    point_gdf = (
        gpd.GeoDataFrame(pd.concat(point_gdfs, ignore_index=True), crs=crs)
        if point_gdfs
        else gpd.GeoDataFrame(geometry=[], crs=crs)
    )

    return {
        "polygon_gdf": polygon_gdf,
        "line_gdf": line_gdf,
        "point_gdf": point_gdf,
    }
