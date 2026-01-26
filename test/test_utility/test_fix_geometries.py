#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from geogenalg.utility import fix_geometries


@pytest.mark.parametrize(
    ("input_geoms", "expected_geoms"),
    [
        ([Point(0, 0), Point(1, 1)], [Point(0, 0), Point(1, 1)]),
        ([Point(0, 0), Point(), Point(2, 2)], [Point(0, 0), Point(2, 2)]),
        ([Point(0, 0), None, Point(1, 1)], [Point(0, 0), Point(1, 1)]),
        ([Point(), None], []),
    ],
    ids=[
        "no empty geometries",
        "one empty geometry",
        "one NaN geometry",
        "all empty",
    ],
)
def test_drop_empty_geometries(
    input_geoms: list[BaseGeometry], expected_geoms: list[BaseGeometry]
):
    gdf = gpd.GeoDataFrame(geometry=input_geoms)
    expected_gdf = gpd.GeoDataFrame(geometry=expected_geoms)
    result = fix_geometries.drop_empty_geometries(gdf).reset_index(drop=True)
    expected_gdf = expected_gdf.reset_index(drop=True)
    assert_frame_equal(result, expected_gdf, check_dtype=False)


def test_fix_invalid_geometries_with_valid_polygon():
    """Valid geometry should remain unchanged."""
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame(geometry=[polygon])

    result_gdf = fix_geometries.fix_invalid_geometries(gdf)

    assert result_gdf.geometry.iloc[0].equals(polygon)
    assert result_gdf.geometry.iloc[0].is_valid


def test_fix_invalid_geometries_with_self_intersection():
    """Self-intersecting polygon should be repaired."""
    invalid_polygon = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame(geometry=[invalid_polygon])

    result_gdf = fix_geometries.fix_invalid_geometries(gdf)
    repaired_geom = result_gdf.geometry.iloc[0]

    assert repaired_geom.is_valid
    assert isinstance(repaired_geom, (Polygon, MultiPolygon))


def test_fix_invalid_geometries_with_empty_geometry():
    """Empty geometry should remain empty and valid."""
    gdf = gpd.GeoDataFrame(geometry=[Polygon(), None])

    result_gdf = fix_geometries.fix_invalid_geometries(gdf)

    assert result_gdf.geometry.is_empty.iloc[0]
    assert result_gdf.geometry.isna().iloc[1]


def test_fix_invalid_geometries_mixed_valid_and_invalid():
    """Only invalid geometries should be changed."""
    valid_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    invalid_polygon = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame(geometry=[valid_polygon, invalid_polygon])

    result_gdf = fix_geometries.fix_invalid_geometries(gdf)

    assert result_gdf.geometry.iloc[0].equals(valid_polygon)
    assert result_gdf.geometry.iloc[0].is_valid

    repaired = result_gdf.geometry.iloc[1]
    assert repaired.is_valid
    assert isinstance(repaired, (Polygon, MultiPolygon))
