#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import re
from typing import Literal

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from numpy.testing import assert_approx_equal
from shapely import box
from shapely.geometry import LineString, Point, Polygon

from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.merge import (
    buffer_and_merge_polygons,
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
    ("input_gdf", "expected_gdf", "by_column", "inherit_from"),
    [
        # 1.
        (
            GeoDataFrame(
                {"id": [1], "group": ["A"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            ),
            GeoDataFrame(
                {
                    "id": [1],
                    "group": ["A"],
                    "old_ids": [(1,)],
                },
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            ),
            "group",
            "min_id",
        ),
        # 2.
        (
            GeoDataFrame(
                {"id": [2, 1], "group": ["A", "A"]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1],
                    "group": ["A"],
                    "old_ids": [(2, 1)],
                },
                geometry=[
                    Polygon(
                        [
                            (0, 1),
                            (1, 1),
                            (2, 1),
                            (2, 0),
                            (1, 0),
                            (0, 0),
                            (0, 1),
                        ]
                    ),
                ],
            ),
            "group",
            "min_id",
        ),
        # 3.
        (
            GeoDataFrame(
                {"id": [1, 2], "group": ["A", "B"]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1, 2],
                    "group": ["A", "B"],
                    "old_ids": [(1,), (2,)],
                },
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                ],
            ),
            "group",
            "min_id",
        ),
        # 4.
        (
            GeoDataFrame(
                {"id": [1, 2], "group": ["A", "B"]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1],
                    "group": ["A"],
                    "old_ids": [(1, 2)],
                },
                geometry=[Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])],
            ),
            None,
            "min_id",
        ),
        # 5.
        (
            GeoDataFrame(
                {"group": ["A", "A"], "id": [1, 2]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            ),
            GeoDataFrame(
                {
                    "group": ["A", "A"],
                    "id": [1, 2],
                    "old_ids": [(1,), (2,)],
                },
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            ),
            "group",
            "min_id",
        ),
        # 6.
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
                    "old_ids": [(1, 2), (3,)],
                },
                geometry=[
                    Polygon(
                        [(0, 0), (2, 0), (2, 1), (3, 1), (3, 3), (1, 3), (1, 2), (0, 2)]
                    ),
                    Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                ],
            ),
            "group",
            "min_id",
        ),
        # 7.
        (
            GeoDataFrame(
                {
                    "group": ["A", "A", "B"],
                    "id": [1, 2, 3],
                    "name": ["first", "second", "third"],
                },
                geometry=[
                    box(0, 0, 2, 2),
                    box(1, 1, 4, 4),
                    box(1, 1, 4, 4),
                ],
            ),
            GeoDataFrame(
                {
                    "group": ["A", "B"],
                    "id": [2, 3],
                    "name": ["second", "third"],
                    "old_ids": [(2, 1), (3,)],
                },
                geometry=[
                    Polygon(
                        [(0, 0), (0, 2), (1, 2), (1, 4), (4, 4), (4, 1), (2, 1), (2, 0)]
                    ),
                    Polygon([(4, 1), (1, 1), (1, 4), (4, 4)]),
                ],
            ),
            "group",
            "most_intersection",
        ),
    ],
    ids=[
        "single polygon",  # 1.
        "_two polygons in same group should dissolve",  # 2.
        "two polygons in different groups should not dissolve",  # 3.
        "two polygons without group parameter should dissolve",  # 4.
        "two separate polygons in same group should not dissolve",  # 5.
        "three intersecting polygons with extra attributes, two in same group",  # 6.
        "three intersecting polygons with extra attributes, two in same group, inherit from most_intersection",  # 7.
    ],
)
def test_dissolve_and_inherit_attributes(
    input_gdf: GeoDataFrame,
    expected_gdf: GeoDataFrame,
    by_column: str | None,
    inherit_from: Literal["min_id", "most_intersection"],
):
    input_gdf = input_gdf.set_index("id")
    expected_gdf = expected_gdf.set_index("id")

    result_gdf = dissolve_and_inherit_attributes(
        input_gdf,
        by_column,
        "old_ids",
        inherit_from,
    )

    assert_geodataframe_equal(
        result_gdf,
        expected_gdf,
        check_like=True,
        normalize=True,
    )


def test_dissolve_and_inherit_attributes_handles_empty_gdf_correctly():
    input_gdf = GeoDataFrame(columns=["id", "group"], geometry=[])
    expected_gdf = GeoDataFrame(columns=["id", "group"], geometry=[])

    result_gdf = dissolve_and_inherit_attributes(
        input_gdf,
        by_column="group",
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
        dissolve_and_inherit_attributes(invalid_gdf, by_column="group")


def test_buffer_and_merge_polygons():
    # CASE 1: Single polygon (nothing to merge)
    polygon = box(0.0, 0.0, 1.0, 1.0)
    gdf = GeoDataFrame(geometry=[polygon], crs="EPSG:3067")

    result = buffer_and_merge_polygons(gdf, buffer_distance=1.0)

    assert len(result) == 1
    assert result.geometry.iloc[0].equals(polygon)

    # CASE 2: Two nearby polygons (merge)
    polygon_1 = box(0.0, 0.0, 1.0, 1.0)
    polygon_2 = box(1.5, 0.0, 2.5, 1.0)
    gdf = GeoDataFrame(geometry=[polygon_1, polygon_2], crs="EPSG:3067")

    result = buffer_and_merge_polygons(gdf, buffer_distance=0.5)

    # Should be merged into a single polygon
    assert len(result) == 1

    merged = result.geometry.iloc[0]
    assert merged.contains(polygon_1.centroid)
    assert merged.contains(polygon_2.centroid)

    # CASE 3: Two distant polygons (no merge)
    polygon_1 = box(0.0, 0.0, 1.0, 1.0)
    polygon_2 = box(4.0, 0.0, 5.0, 1.0)
    gdf = GeoDataFrame(geometry=[polygon_1, polygon_2], crs="EPSG:3067")

    result = buffer_and_merge_polygons(gdf, buffer_distance=1.0)

    # Polygons should stay separate
    assert len(result) == 2
    assert_approx_equal(polygon_1.area, result.geometry.iloc[0].area, significant=3)
    assert_approx_equal(polygon_2.area, result.geometry.iloc[1].area, significant=3)


def test_buffer_and_merge_polygons_invalid_geometry_type():
    with pytest.raises(
        GeometryTypeError,
        match=re.escape(
            "Buffer and merge polygons works only with (Multi)Polygon geometries."
        ),
    ):
        buffer_and_merge_polygons(
            input_gdf=GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3067"),
            buffer_distance=1.0,
        )
