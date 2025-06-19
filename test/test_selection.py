#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd
import pytest
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
    ("lines_gdf", "polygons_gdf", "expected_indices"),
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
                geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])], crs="EPSG:3067"
            ),
            {1, 2},
        ),
        (
            gpd.GeoDataFrame(
                {"id": [0]},
                geometry=[LineString([(0, 0), (1, 1)])],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(geometry=[], crs="EPSG:3067"),
            {0},
        ),
    ],
    ids=["one_line_on_edge_two_not_on_edge", "empty_polygon_boundary_none"],
)
def test_remove_lines_on_polygon_edges(
    lines_gdf: gpd.GeoDataFrame, polygons_gdf: gpd.GeoDataFrame, expected_indices: dict
):
    result = selection.remove_lines_on_polygon_edges(lines_gdf, polygons_gdf)

    assert set(result["id"]) == set(expected_indices)
