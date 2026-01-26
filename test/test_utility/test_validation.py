#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
from geopandas import GeoDataFrame, GeoSeries
from shapely import Point, box
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

from geogenalg.utility.validation import (
    ShapelyGeometryTypeString,
    check_gdf_geometry_type,
    check_geoseries_geometry_type,
    geometry_string_to_type,
)


@pytest.mark.parametrize(
    ("gdf", "accepted_types", "expected"),
    [
        (GeoDataFrame(geometry=[Point(0, 0), Point(0, 1)]), ["Polygon"], False),
        (GeoDataFrame(geometry=[Point(0, 0), box(0, 0, 1, 1)]), ["Polygon"], False),
        (
            GeoDataFrame(geometry=[Point(0, 0), box(0, 0, 1, 1)]),
            ["Polygon", "Point"],
            True,
        ),
    ],
    ids=[
        "all wrong",
        "one wrong",
        "all correct",
    ],
)
def test_check_gdf_geometry_type(
    gdf: GeoDataFrame,
    accepted_types: list[str],
    expected: bool,
):
    assert check_gdf_geometry_type(gdf, accepted_types) == expected


@pytest.mark.parametrize(
    ("geoseries", "accepted_types", "expected"),
    [
        (GeoSeries([Point(0, 0), Point(0, 1)]), ["Polygon"], False),
        (GeoSeries([Point(0, 0), box(0, 0, 1, 1)]), ["Polygon"], False),
        (GeoSeries([Point(0, 0), box(0, 0, 1, 1)]), ["Polygon", "Point"], True),
    ],
    ids=[
        "all wrong",
        "one wrong",
        "all correct",
    ],
)
def test_check_geoseries_geometry_type(
    geoseries: GeoSeries,
    accepted_types: list[str],
    expected: bool,
):
    assert check_geoseries_geometry_type(geoseries, accepted_types) == expected


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("Point", Point),
        ("LineString", LineString),
        ("LinearRing", LinearRing),
        ("Polygon", Polygon),
        ("MultiPoint", MultiPoint),
        ("MultiLineString", MultiLineString),
        ("MultiPolygon", MultiPolygon),
        ("GeometryCollection", GeometryCollection),
    ],
    ids=[
        "Point",
        "LineString",
        "LinearRing",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
        "GeometryCollection",
    ],
)
def test_geometry_string_to_type(
    string: ShapelyGeometryTypeString,
    expected: type[BaseGeometry],
):
    assert geometry_string_to_type(string) is expected
