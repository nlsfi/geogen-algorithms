#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from pathlib import Path

from geopandas import GeoDataFrame, read_file
from pandas import concat


def read_gdf_from_file_and_set_index(
    file_name: str | Path,
    unique_id_column: str,
    *,
    layer: str | None = None,
) -> GeoDataFrame:
    """Read GeoDataFrame from file and set its index to a given column.

    Args:
    ----
        file_name: Path to dataset file to read into a GeoDataFrame.
        unique_id_column: Column in the dataset containing unique identifiers.
        layer: Allows optionally specifying layer if dataset contains multiple.

    Returns:
    -------
        GeoDataFrame read from the file, with its index set to the column and
        the index name set to the given column name.

    """
    gdf = read_file(file_name, layer=layer)
    gdf = gdf.set_index(unique_id_column)
    gdf.index.name = unique_id_column

    return gdf


def group_gdfs_by_geometry_type(
    input_gdfs: list[GeoDataFrame],
) -> dict[str, GeoDataFrame]:
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
        if not isinstance(input_gdf, GeoDataFrame):
            continue
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
        GeoDataFrame(concat(polygon_gdfs, ignore_index=True), crs=crs)
        if polygon_gdfs
        else GeoDataFrame(geometry=[], crs=crs)
    )
    line_gdf = (
        GeoDataFrame(concat(line_gdfs, ignore_index=True), crs=crs)
        if line_gdfs
        else GeoDataFrame(geometry=[], crs=crs)
    )
    point_gdf = (
        GeoDataFrame(concat(point_gdfs, ignore_index=True), crs=crs)
        if point_gdfs
        else GeoDataFrame(geometry=[], crs=crs)
    )

    return {
        "polygon_gdf": polygon_gdf,
        "line_gdf": line_gdf,
        "point_gdf": point_gdf,
    }
