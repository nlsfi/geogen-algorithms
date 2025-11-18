#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import re

import pytest
from geopandas import GeoDataFrame
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString, Point, Polygon

from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.merge import (
    dissolve_and_inherit_attributes,
    merge_connecting_lines_by_attribute,
)


@pytest.mark.parametrize(
    (
        "input_gdf",
        "attribute",
        "expected_num_geoms",
        "expected_attribute_values",
        "expected_geometries",
        "expected_old_ids",
    ),
    [
        (
            GeoDataFrame(
                {
                    "class": ["a", "a", "b"],
                    "id": [1, 2, 3],
                },
                geometry=[
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(10, 10), (11, 11)]),
                ],
                crs="EPSG:3067",
            ),
            "class",
            2,
            {"a", "b"},
            [
                LineString([(0, 0), (1, 1), (2, 2)]),
                LineString([(10, 10), (11, 11)]),
            ],
            [["1", "2"], ["3"]],
        ),
        (
            GeoDataFrame(
                {
                    "class": ["b", "b"],
                    "id": [1, 2],
                },
                geometry=[
                    LineString([(0, 0), (1, 1)]),
                    LineString([(2, 1), (2, 2)]),
                ],
                crs="EPSG:3067",
            ),
            "class",
            2,
            {"b"},
            [
                LineString([(0, 0), (1, 1)]),
                LineString([(2, 1), (2, 2)]),
            ],
            [["1"], ["2"]],
        ),
        (
            GeoDataFrame(
                {
                    "class": ["c", "c", "c"],
                    "id": [1, 2, 3],
                },
                geometry=[
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(2, 2), (3, 3)]),
                ],
                crs="EPSG:3067",
            ),
            "class",
            1,
            {"c"},
            [
                LineString([(0, 0), (1, 1), (2, 2), (3, 3)]),
            ],
            [["1", "2", "3"]],
        ),
        (
            GeoDataFrame(
                {
                    "class": ["x", "y"],
                    "id": [1, 2],
                },
                geometry=[
                    LineString([(0, 0), (1, 0)]),
                    LineString([(10, 10), (11, 11)]),
                ],
                crs="EPSG:3067",
            ),
            "class",
            2,
            {"x", "y"},
            [
                LineString([(0, 0), (1, 0)]),
                LineString([(10, 10), (11, 11)]),
            ],
            [["1"], ["2"]],
        ),
    ],
    ids=[
        "merge_two_lines",
        "disjoint_lines_same_class",
        "merge_three_lines",
        "disjoint_lines_different_classes",
    ],
)
def test_merge_lines_with_same_attribute_value_into_one(  # noqa: PLR0917
    input_gdf: GeoDataFrame,
    attribute: str,
    expected_num_geoms: int,
    expected_attribute_values: set[str],
    expected_geometries: list[LineString],
    expected_old_ids: list[list[str]],
):
    input_gdf = input_gdf.set_index("id")
    input_gdf.index = input_gdf.index.astype("string")
    result_gdf = merge_connecting_lines_by_attribute(input_gdf, attribute)
    assert len(result_gdf) == expected_num_geoms
    assert set(result_gdf[attribute]) == expected_attribute_values
    assert all(geom.geom_type == "LineString" for geom in result_gdf.geometry)
    assert all(
        any(result_geom.equals(expected_geom) for result_geom in result_gdf.geometry)
        for expected_geom in expected_geometries
    )
    old_ids = [sorted(ids) for ids in result_gdf["old_ids"]]
    assert old_ids == expected_old_ids


@pytest.mark.parametrize(
    (
        "input_gdf",
        "expected_gdf",
    ),
    [
        (
            GeoDataFrame(
                {"id": [1], "group": ["A"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            ),
            GeoDataFrame(
                {"id": [1], "group": ["A"], "dissolve_members": [None]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [2, 1], "group": ["A", "A"]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                ],
            ),
            GeoDataFrame(
                {"id": [1], "group": ["A"], "dissolve_members": [["1", "2"]]},
                geometry=[Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [1, 2], "group": ["A", "B"]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                ],
            ),
            GeoDataFrame(
                {"id": [1, 2], "group": ["A", "B"], "dissolve_members": [None, None]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {"group": ["A", "A"], "id": [1, 2]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            ),
            GeoDataFrame(
                {"group": ["A", "A"], "id": [1, 2], "dissolve_members": [None, None]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "group": ["A", "A", "B"],
                    "id": [1, 2, 3],
                    "name": ["first", "second", "third"],
                },
                geometry=[
                    Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                    Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                    Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                ],
            ),
            GeoDataFrame(
                {
                    "group": ["A", "B"],
                    "id": [1, 3],
                    "name": ["first", "third"],
                    "dissolve_members": [["1", "2"], None],
                },
                geometry=[
                    Polygon(
                        [(0, 0), (2, 0), (2, 1), (3, 1), (3, 3), (1, 3), (1, 2), (0, 2)]
                    ),
                    Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2],
                    "group": ["A", "A"],
                    "dissolve_members": [["10", "1"], ["20"]],
                },
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1],
                    "group": ["A"],
                    "dissolve_members": [["1", "10", "2", "20"]],
                },
                geometry=[Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2, 3],
                    "group": ["A", "A", "A"],
                    "dissolve_members": [None, ["10", "20"], ["3", "30"]],
                },
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(0.9, 0), (2, 0), (2, 1), (0.9, 1)]),
                    Polygon([(1.9, 0), (3, 0), (3, 1), (1.9, 1)]),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1],
                    "group": ["A"],
                    "dissolve_members": [["1", "10", "2", "20", "3", "30"]],
                },
                geometry=[Polygon([(0, 0), (3, 0), (3, 1), (0, 1)])],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "group": ["A", "A", "B", "B"],
                    "dissolve_members": [["100"], None, ["3", "200", "201"], None],
                },
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                    Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
                    Polygon([(1, 2), (2, 2), (2, 3), (1, 3)]),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1, 3],
                    "group": ["A", "B"],
                    "dissolve_members": [["1", "100", "2"], ["200", "201", "3", "4"]],
                },
                geometry=[
                    Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]),
                    Polygon([(0, 2), (2, 2), (2, 3), (0, 3)]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2, 3],
                    "group": ["A", "A", "A"],
                    "dissolve_members": [["1", "5"], None, ["3", "50", "789"]],
                },
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(0.9, 0), (2, 0), (2, 1), (0.9, 1)]),
                    Polygon([(2.1, 0), (3, 0), (3, 1), (2.1, 1)]),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1, 3],
                    "group": ["A", "A"],
                    "dissolve_members": [["1", "2", "5"], ["3", "50", "789"]],
                },
                geometry=[
                    Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]),
                    Polygon([(2.1, 0), (3, 0), (3, 1), (2.1, 1)]),
                ],
            ),
        ),
    ],
    ids=[
        "single polygon",
        "two polygons in same group should dissolve",
        "two polygons in different groups should not dissolve",
        "two separate polygons in same group should not dissolve",
        "three intersecting polygons with extra attributes, two in same group",
        "polygons already have attribute for dissolve_members",
        "mixed None and lists in same group",
        "multiple groups with mixed dissolve_members",
        "two dissolving geometries and one separate in same group",
    ],
)
def test_dissolve_and_inherit_attributes(
    input_gdf: GeoDataFrame,
    expected_gdf: GeoDataFrame,
):
    result_gdf = dissolve_and_inherit_attributes(
        input_gdf,
        by_column="group",
        unique_key_column="id",
        dissolve_members_column="dissolve_members",
    )

    result_sorted = result_gdf.sort_values("id").reset_index(drop=True)
    expected_sorted = expected_gdf.sort_values("id").reset_index(drop=True)

    # Check geometries
    for result_geom, expected_geom in zip(
        result_sorted.geometry, expected_sorted.geometry, strict=False
    ):
        assert result_geom.equals(expected_geom), (
            f"Geometries differ: {result_geom} vs {expected_geom}"
        )

    # Check attributes
    result_attrs = result_sorted.drop(columns="geometry")
    expected_attrs = expected_sorted.drop(columns="geometry")
    assert_frame_equal(result_attrs, expected_attrs)


def test_dissolve_and_inherit_attributes_handles_empty_gdf_correctly():
    input_gdf = GeoDataFrame(columns=["id", "group"], geometry=[])
    expected_gdf = GeoDataFrame(columns=["id", "group"], geometry=[])

    result_gdf = dissolve_and_inherit_attributes(
        input_gdf,
        by_column="group",
        unique_key_column="id",
        dissolve_members_column="dissolve_members",
    )

    assert result_gdf.equals(expected_gdf)


def test_dissolve_and_inherit_attributes_raises_on_non_polygon_geometry():
    invalid_gdf = GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ],
    )

    with pytest.raises(
        GeometryTypeError,
        match=re.escape("Dissolve only supports Polygon or MultiPolygon geometries."),
    ):
        dissolve_and_inherit_attributes(
            invalid_gdf, by_column="group", unique_key_column="id"
        )
