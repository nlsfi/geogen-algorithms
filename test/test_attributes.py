#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from shapely import box
from shapely.geometry import LineString, Polygon

from geogenalg.attributes import inherit_attributes, inherit_attributes_from_largest


@pytest.mark.parametrize(
    ("source_gdf", "target_gdf", "expected_attribute_value", "expected_geometry"),
    [
        (
            GeoDataFrame(
                {"id": [1], "mtk_id": ["3491230"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                crs="EPSG:3067",
            ),
            GeoDataFrame(
                {"id": [23], "mtk_id": ["009312"], "class": ["cultivated_land"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                crs="EPSG:3067",
            ),
            "3491230",
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2, 3],
                    "mtk_id": ["3491230", "344531", "1234"],
                    "class": ["park", "park", "park"],
                },
                geometry=[
                    Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 1)]),
                    Polygon([(1, 1), (1.5, 1), (1.5, 1.5), (1, 1.5)]),
                    Polygon([(0.9, 0.9), (1.9, 0.9), (1.9, 1.9), (0.9, 1.9)]),
                ],
                crs="EPSG:3067",
            ),
            GeoDataFrame(
                {"id": [4], "mtk_id": ["0000"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                crs="EPSG:3067",
            ),
            ["3491230", "1234"],
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ),
        (
            GeoDataFrame(
                {"id": [1], "mtk_id": ["888888"]},
                geometry=[Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])],
                crs="EPSG:3067",
            ),
            GeoDataFrame(
                {"id": [99], "mtk_id": ["555666"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                crs="EPSG:3067",
            ),
            "888888",
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2, 3],
                    "mtk_id": ["123", "456", "789"],
                },
                geometry=[
                    LineString([(0, 0), (0.5, 0)]),
                    LineString([(0.5, 0), (1, 0)]),
                    LineString([(12, 3), (13, 4)]),
                ],
                crs="EPSG:3067",
            ),
            GeoDataFrame(
                {"id": [42], "class": ["fence"]},
                geometry=[LineString([(0, 0), (1, 0)])],
                crs="EPSG:3067",
            ),
            ["123", "456"],
            LineString([(0, 0), (1, 0)]),
        ),
    ],
    ids=[
        "same_polygon",
        "two_partly_intersecting_polygons",
        "disjoint_polygon_geometries",
        "two_partly_intersecting_linestrings",
    ],
)
def test_inherit_attributes(
    source_gdf: GeoDataFrame,
    target_gdf: GeoDataFrame,
    expected_attribute_value: str,
    expected_geometry: Polygon | LineString,
):
    result_gdf = inherit_attributes(source_gdf, target_gdf)
    assert "mtk_id" in result_gdf.columns
    assert result_gdf.iloc[0]["mtk_id"] in expected_attribute_value
    assert result_gdf.iloc[0].geometry.equals(expected_geometry)
    assert len(result_gdf) == len(target_gdf)
    assert set(source_gdf.columns) == set(result_gdf.columns)


@pytest.mark.parametrize(
    ("source_gdf", "target_gdf", "expected_gdf"),
    [
        (
            GeoDataFrame(
                {
                    "id": [1, 2, 3],
                    "other": [111111, 222222, 333333],
                },
                geometry=[
                    box(0, 0, 1, 1),
                    box(5, 5, 6, 6),
                    box(9, 9, 10, 10),
                ],
            ),
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 1]]),
                    LineString([[5, 5], [6, 6]]),
                    LineString([[9, 9], [10, 10]]),
                ]
            ),
            GeoDataFrame(
                {
                    "id": [1, 2, 3],
                    "other": [111111, 222222, 333333],
                    "__old_ids": [
                        (1,),
                        (2,),
                        (3,),
                    ],
                },
                geometry=[
                    LineString([[0, 0], [1, 1]]),
                    LineString([[5, 5], [6, 6]]),
                    LineString([[9, 9], [10, 10]]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2, 3],
                    "other": [111111, 222222, 333333],
                },
                geometry=[
                    box(0, 0, 1, 1),
                    box(5, 5, 6, 6),
                    box(6, 6, 8, 4),
                ],
            ),
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 1]]),
                    LineString([[5.5, 5.5], [8, 5.5]]),
                ]
            ),
            GeoDataFrame(
                {
                    "id": [1, 3],
                    "other": [111111, 333333],
                    "__old_ids": [
                        (1,),
                        (3, 2),
                    ],
                },
                geometry=[
                    LineString([[0, 0], [1, 1]]),
                    LineString([[5.5, 5.5], [8, 5.5]]),
                ],
            ),
        ),
    ],
    ids=[
        "no multiple intersections",
        "some multiple intersections",
    ],
)
def test_inherit_attributes_from_largest(
    source_gdf: GeoDataFrame,
    target_gdf: GeoDataFrame,
    expected_gdf: GeoDataFrame,
):
    source_gdf = source_gdf.set_index("id")
    expected_gdf = expected_gdf.set_index("id")
    result = inherit_attributes_from_largest(
        source_gdf, target_gdf, old_ids_column="__old_ids"
    )
    assert_geodataframe_equal(
        result,
        expected_gdf,
        check_like=True,
    )
