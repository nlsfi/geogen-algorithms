#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal
from shapely import MultiLineString
from shapely.geometry import LineString, Point, Polygon

from geogenalg import selection


@pytest.mark.parametrize(
    ("input_lines", "threshold", "expected_num_lines"),
    [
        ([LineString([(0, 0), (5, 0)])], 2.0, 1),
        ([LineString([(0, 0), (1.9, 0)])], 2.0, 0),
        (
            [
                LineString([(0, 0), (2, 0)]),
                LineString([(2, 0), (2.5, 1)]),
                LineString([(2.5, 1), (5.5, 0)]),
            ],
            5.0,
            1,
        ),
        (
            [
                LineString([(0, 0), (2, 0)]),
                LineString([(2, 0), (2.5, 1)]),
                LineString([(2.5, 1), (2.5, 2)]),
            ],
            2.0,
            2,
        ),
    ],
    ids=[
        "line_length_over_threshold",
        "line_length_under_threshold",
        "three_connected_lines_two_removed",
        "three_connected_lines_one_removed",
    ],
)
def test_remove_short_lines(
    input_lines: list[LineString], threshold: float, expected_num_lines: int
):
    lines_gdf = gpd.GeoDataFrame(
        {
            "id": enumerate(input_lines),
        },
        geometry=input_lines,
        crs="EPSG:3067",
    )
    result = selection.remove_disconnected_short_lines(lines_gdf, threshold)
    assert len(result) == expected_num_lines


@pytest.mark.parametrize(
    (
        "polygon_gdf",
        "point_gdf",
        "expected_containing_point",
        "expected_not_containing_point",
    ),
    [
        (
            gpd.GeoDataFrame(
                {"id": [0, 1, 2]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                    Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
                ],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(
                {"point_id": [0, 1]},
                geometry=[Point(2.5, 0.5), Point(4.5, 0.5)],
                crs="EPSG:3067",
            ),
            {1, 2},
            {0},
        ),
        (
            gpd.GeoDataFrame(
                {"id": [0, 1]},
                geometry=[
                    Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                    Polygon([(3, 0), (5, 0), (5, 2), (3, 2)]),
                ],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(
                {"point_id": []},
                geometry=[],
                crs="EPSG:3067",
            ),
            set(),
            {0, 1},
        ),
    ],
    ids=[
        "two_polygons_contain_points_one_not",
        "no_polygons_contain_points",
    ],
)
def test_split_polygons_by_point_containment(
    polygon_gdf: gpd.GeoDataFrame,
    point_gdf: gpd.GeoDataFrame,
    expected_containing_point: set[int],
    expected_not_containing_point: set[int],
):
    with_points, without_points = selection.split_polygons_by_point_intersection(
        polygon_gdf, point_gdf
    )

    assert set(with_points["id"]) == expected_containing_point
    assert set(without_points["id"]) == expected_not_containing_point


@pytest.mark.parametrize(
    ("lines_gdf", "polygons_gdf", "expected_gdf"),
    [
        (
            gpd.GeoDataFrame(
                {"id": [0, 1, 2]},
                geometry=[
                    LineString([(0, 0), (2, 0)]),
                    LineString([(0.5, 0.5), (1.5, 1.5)]),
                    LineString([(3, 3), (4, 4)]),
                ],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(
                geometry=[
                    Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                    Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
                    Polygon([(0, 0), (-2, 0), (-2, -2), (0, -2)]),
                ],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(
                {"id": [1, 2]},
                geometry=[
                    LineString([(0.5, 0.5), (1.5, 1.5)]),
                    LineString([(3, 3), (4, 4)]),
                ],
                crs="EPSG:3067",
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [0]},
                geometry=[LineString([(0, 0), (1, 1)])],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(geometry=[], crs="EPSG:3067"),
            gpd.GeoDataFrame(
                {"id": [0]},
                geometry=[LineString([(0, 0), (1, 1)])],
                crs="EPSG:3067",
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [0]},
                geometry=[LineString([(1, -1), (1, 3)])],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(
                geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(
                {"id": [0]},
                geometry=[
                    MultiLineString(
                        [
                            [(1, -1), (1, 0)],
                            [(1, 0), (1, 2)],
                            [(1, 2), (1, 3)],
                        ]
                    )
                ],
                crs="EPSG:3067",
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [0, 1]},
                geometry=[
                    LineString([(0, 0), (1, 0), (1, 1)]),
                    LineString([(0, 0), (0, 1), (-1, 1)]),
                ],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(
                geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(
                {"id": [0, 1]},
                geometry=[
                    LineString([(1, 0), (1, 1)]),
                    LineString([(0, 1), (-1, 1)]),
                ],
                crs="EPSG:3067",
            ),
        ),
    ],
    ids=[
        "one_line_on_edge_two_not_on_edge",
        "empty_polygon_boundary_none",
        "line_intersects_boundary_at_one_point",
        "two_lines_partially_on_boundary",
    ],
)
def test_remove_parts_of_lines_on_polygon_edges(
    lines_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    expected_gdf: gpd.GeoDataFrame,
):
    result_gdf = selection.remove_parts_of_lines_on_polygon_edges(
        lines_gdf, polygons_gdf
    )

    assert_frame_equal(
        result_gdf.sort_index().reset_index(drop=True),
        expected_gdf.sort_index().reset_index(drop=True),
        check_like=True,
    )


@pytest.mark.parametrize(
    ("input_polygons", "threshold", "expected_polygons"),
    [
        (
            [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            2,
            [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        ),
        (
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            ],
            0.1,
            [],
        ),
        (
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
                Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
            ],
            1,
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
            ],
        ),
    ],
    ids=[
        "one_polygon_under_threshold",
        "two_polygons_over_threshold",
        "two_polygons_under_and_one_over_threshold",
    ],
)
def test_remove_large_polygons_correct(
    input_polygons: list[Polygon], threshold: float, expected_polygons: list[Polygon]
):
    input_gdf = gpd.GeoDataFrame(geometry=input_polygons)
    expected_gdf = gpd.GeoDataFrame(geometry=expected_polygons)

    result_gdf = selection.remove_large_polygons(input_gdf, threshold)

    assert_frame_equal(
        result_gdf.sort_index().reset_index(drop=True),
        expected_gdf.sort_index().reset_index(drop=True),
        check_like=True,
    )
