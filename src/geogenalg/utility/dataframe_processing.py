#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, Unpack

from geopandas import GeoDataFrame, read_file
from geopandas.geoseries import GeoSeries
from pandas import Series, concat

from geogenalg.core.exceptions import GeoCombineError


class ConcatParameters(TypedDict):
    """Types for pandas.concat."""

    axis: NotRequired[Literal[0, 1]]
    join: NotRequired[Literal["inner", "outer"]]
    ignore_index: NotRequired[bool]
    verify_integrity: NotRequired[bool]
    sort: NotRequired[bool]
    copy: NotRequired[bool]


def copy_gdf_as_empty(
    input_gdf: GeoDataFrame,
    *,
    add_columns: dict[str | None, str] | None = None,
) -> GeoDataFrame:
    """Copy GeoDataFrame, retaining its structure (columns, crs) but not any rows.

    Args:
    ----
        input_gdf: GeoDataFrame to copy.
        add_columns: Optionally define new columns to add to copied GeoDataFrame.
            Key is column name, value is column dtype. If key is None, it will
            not be added.

    Returns:
    -------
        GeoDataFrame with input_gdf's columns and crs but without any rows.

    """
    if add_columns is None:
        add_columns = {}

    # If input_gdf does not have an active geometry column, a DataFrame would
    # be returned. Wrap in gdf constructor to ensure a GeoDataFrame is always
    # returned.
    gdf = GeoDataFrame(input_gdf.iloc[0:0].copy())
    for name, dtype in add_columns.items():
        if name is None:
            continue

        gdf[name] = Series(dtype=dtype)
    return gdf


def add_columns_to_gdf(
    input_gdf: GeoDataFrame,
    add_columns: dict[str | None, str],
) -> GeoDataFrame:
    """Add columns to a GeoDataFrame.

    Args:
    ----
        input_gdf: GeoDataFrame to add to.
        add_columns: New columns to add to GeoDataFrame. Key is column name,
            value is column dtype. If key is None, the column will not be added.

    Returns:
    -------
        GeoDataFrame with added columns.

    """
    columns = {key: value for key, value in add_columns.items() if key is not None}

    if not columns:
        return input_gdf

    gdf = input_gdf.copy()
    for name, dtype in columns.items():
        if name is None:
            continue

        gdf[name] = Series(dtype=dtype)

    return gdf


def _combine_geo_objects(
    geo_objects: Iterable[GeoDataFrame] | Iterable[GeoSeries],
    object_type: type[GeoDataFrame | GeoSeries],
    **kwargs: Unpack[ConcatParameters],
) -> GeoDataFrame | GeoSeries:
    crs = set()
    geom_names = set()

    geo_objects = list(geo_objects)

    if len(geo_objects) == 0:
        msg = "Nothing to combine."
        raise GeoCombineError(msg)

    for geo_object in geo_objects:
        if not isinstance(geo_object, object_type):
            msg = f"Non-{object_type.__name__} object found."
            raise GeoCombineError(msg)

        if isinstance(geo_object, GeoDataFrame) and not geo_object.active_geometry_name:
            msg = "GeoDataFrame does not have active geometry column set."
            raise GeoCombineError(msg)

        geom_names.add(geo_object.geometry.name)
        crs.add(geo_object.crs)

    if len(geo_objects) == 1:
        return geo_objects[0]

    if len(crs) != 1:
        msg = "Different CRSs found in GeoDataFrames."
        raise GeoCombineError(msg)

    if len(geom_names) != 1:
        msg = "Different geometry column names found in GeoDataFrames."
        raise GeoCombineError(msg)

    return concat(geo_objects, **kwargs)


def combine_gdfs(
    gdfs: Iterable[GeoDataFrame],
    **kwargs: Unpack[ConcatParameters],
) -> GeoDataFrame:
    """Combine GeoDataFrames.

    This is a wrapper for pandas.concat() with additional checks that make
    noticing errors easier.

    If single GeoDataFrame is passed, it is returned unchanged.

    Returns
    -------
        Combined GeoDataFrames.

    Raises
    ------
        GeoCombineError: If any check fails.

    """
    combined = _combine_geo_objects(
        gdfs,
        GeoDataFrame,
        **kwargs,
    )

    if not isinstance(combined, GeoDataFrame):
        msg = "Non-GeoDataFrame returned by concat."
        raise GeoCombineError(msg)

    return combined


def combine_geoseries(
    geoseries: Iterable[GeoSeries],
    **kwargs: Unpack[ConcatParameters],
) -> GeoSeries:
    """Combine GeoSeries.

    This is a wrapper for pandas.concat() with additional checks that make
    noticing errors easier.

    If single GeoSeries is passed, it is returned unchanged.

    Returns
    -------
        Combined GeoSeries.

    Raises
    ------
        GeoCombineError: If any check fails.

    """
    combined = _combine_geo_objects(
        geoseries,
        GeoSeries,
        **kwargs,
    )

    if not isinstance(combined, GeoSeries):
        msg = "Non-GeoSeries returned by concat."
        raise GeoCombineError(msg)

    return combined


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
        GeoDataFrame(combine_gdfs(polygon_gdfs, ignore_index=True), crs=crs)
        if polygon_gdfs
        else GeoDataFrame(geometry=[], crs=crs)
    )
    line_gdf = (
        GeoDataFrame(combine_gdfs(line_gdfs, ignore_index=True), crs=crs)
        if line_gdfs
        else GeoDataFrame(geometry=[], crs=crs)
    )
    point_gdf = (
        GeoDataFrame(combine_gdfs(point_gdfs, ignore_index=True), crs=crs)
        if point_gdfs
        else GeoDataFrame(geometry=[], crs=crs)
    )

    return {
        "polygon_gdf": polygon_gdf,
        "line_gdf": line_gdf,
        "point_gdf": point_gdf,
    }
