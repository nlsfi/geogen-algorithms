#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import re

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal
from shapely import MultiLineString
from shapely.geometry import LineString, Point, Polygon

from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.selection import (
    reduce_nearby_points_by_selecting,
    remove_disconnected_short_lines,
    remove_large_polygons,
    remove_parts_of_lines_on_polygon_edges,
    remove_small_holes,
    remove_small_polygons,
    split_polygons_by_point_intersection,
)


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
    result = remove_disconnected_short_lines(lines_gdf, threshold)
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
    with_points, without_points = split_polygons_by_point_intersection(
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
    result_gdf = remove_parts_of_lines_on_polygon_edges(lines_gdf, polygons_gdf)

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
        (
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
            ],
            0.5,
            [],
        ),
    ],
    ids=[
        "one_polygon_under_threshold",
        "two_polygons_over_threshold",
        "two_polygons_under_and_one_over_threshold",
        "no_polygons_under_threshold",
    ],
)
def test_remove_large_polygons_correct(
    input_polygons: list[Polygon], threshold: float, expected_polygons: list[Polygon]
):
    input_gdf = gpd.GeoDataFrame(geometry=input_polygons, crs="EPSG:3067")
    expected_gdf = gpd.GeoDataFrame(geometry=expected_polygons)

    result_gdf = remove_large_polygons(input_gdf, threshold)

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
            0.5,
            [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        ),
        (
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            ],
            5,
            [],
        ),
        (
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
                Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
            ],
            2,
            [
                Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
            ],
        ),
        (
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
            ],
            10,
            [],
        ),
    ],
    ids=[
        "one_polygon_over_threshold",
        "two_polygons_under_threshold",
        "two_polygons_under_and_one_over_threshold",
        "no_polygons_over_threshold",
    ],
)
def test_remove_small_polygons_correct(
    input_polygons: list[Polygon], threshold: float, expected_polygons: list[Polygon]
):
    input_gdf = gpd.GeoDataFrame(geometry=input_polygons, crs="EPSG:3067")
    expected_gdf = gpd.GeoDataFrame(geometry=expected_polygons)

    result_gdf = remove_small_polygons(input_gdf, threshold)

    assert_frame_equal(
        result_gdf.sort_index().reset_index(drop=True),
        expected_gdf.sort_index().reset_index(drop=True),
        check_like=True,
    )


@pytest.mark.parametrize(
    ("input_polygons", "hole_threshold", "expected_polygons"),
    [
        (
            [
                Polygon(
                    shell=[(0, 0), (4, 0), (4, 4), (0, 4)],
                    holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
                )
            ],
            1,
            [
                Polygon(
                    shell=[(0, 0), (4, 0), (4, 4), (0, 4)],
                    holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
                )
            ],
        ),
        (
            [
                Polygon(
                    shell=[(0, 0), (4, 0), (4, 4), (0, 4)],
                    holes=[[(1, 1), (1.2, 1), (1.2, 1.2), (1, 1.2)]],
                )
            ],
            0.1,
            [
                Polygon(
                    shell=[(0, 0), (4, 0), (4, 4), (0, 4)],
                    holes=[],
                )
            ],
        ),
        (
            [
                Polygon(
                    shell=[(0, 0), (5, 0), (5, 5), (0, 5)],
                    holes=[
                        [(1, 1), (1.2, 1), (1.2, 1.2), (1, 1.2)],
                        [(2, 2), (4, 2), (4, 4), (2, 4)],
                    ],
                )
            ],
            0.5,
            [
                Polygon(
                    shell=[(0, 0), (5, 0), (5, 5), (0, 5)],
                    holes=[
                        [(2, 2), (4, 2), (4, 4), (2, 4)],
                    ],
                )
            ],
        ),
        (
            [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            0.1,
            [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        ),
    ],
    ids=[
        "one_large_hole_retained",
        "one_small_hole_removed",
        "mixed_holes_filtering",
        "no_holes",
    ],
)
def test_remove_small_holes_correct(
    input_polygons: list[Polygon],
    hole_threshold: float,
    expected_polygons: list[Polygon],
):
    input_gdf = gpd.GeoDataFrame(geometry=input_polygons)
    expected_gdf = gpd.GeoDataFrame(geometry=expected_polygons)

    result_gdf = remove_small_holes(input_gdf, hole_threshold)

    assert_frame_equal(
        result_gdf.sort_index().reset_index(drop=True),
        expected_gdf.sort_index().reset_index(drop=True),
        check_like=True,
    )


@pytest.mark.parametrize(
    (
        "input_gdf",
        "reference_gdf",
        "distance_threshold",
        "expected_gdf",
    ),
    [
        (
            gpd.GeoDataFrame(
                {"id": [1, 2], "priority": [10, 20]},
                geometry=[Point(0, 0), Point(10, 0)],
            ),
            None,
            5.0,
            gpd.GeoDataFrame(
                {"id": [1, 2], "priority": [10, 20]},
                geometry=[Point(0, 0), Point(10, 0)],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [1, 2], "priority": [10, 20]},
                geometry=[Point(0, 0), Point(1, 0)],
            ),
            None,
            2.0,
            gpd.GeoDataFrame(
                {"id": [2], "priority": [20]},
                geometry=[Point(1, 0)],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [1], "priority": [10]},
                geometry=[Point(0, 0)],
            ),
            gpd.GeoDataFrame(
                {"id": [99], "priority": [5]},
                geometry=[Point(1, 0)],
            ),
            2.0,
            gpd.GeoDataFrame(
                {"id": [1], "priority": [10]},
                geometry=[Point(0, 0)],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [1], "priority": [10]},
                geometry=[Point(0, 0)],
            ),
            gpd.GeoDataFrame(
                {"id": [2], "priority": [10]},
                geometry=[Point(1, 0)],
            ),
            2.0,
            gpd.GeoDataFrame(columns=["id", "priority"], geometry=[]),
        ),
        (
            gpd.GeoDataFrame(columns=["id", "priority"], geometry=[]),
            None,
            2.0,
            gpd.GeoDataFrame(columns=["id", "priority"], geometry=[]),
        ),
        (
            gpd.GeoDataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "priority": [1, 2, 3, 4, 5],
                    "name": ["A", "B", "C", "D", "E"],
                },
                geometry=[Point(x, 0) for x in range(5)],
            ),
            None,
            1.5,
            gpd.GeoDataFrame(
                {
                    "id": [5],
                    "priority": [5],
                    "name": ["E"],
                },
                geometry=[Point(4, 0)],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [1], "priority": [5]},
                geometry=[Point(0, 0)],
            ),
            gpd.GeoDataFrame(
                {
                    "id": [1, 99],
                    "priority": [5, 10],
                },
                geometry=[Point(0, 0), Point(1, 0)],
            ),
            2.0,
            gpd.GeoDataFrame(columns=["id", "priority"], geometry=[]),
        ),
        (
            gpd.GeoDataFrame(
                {"id": list(range(1, 11)), "priority": [1] * 9 + [100]},
                geometry=[
                    Point(5, 0),
                    Point(3.5, 3.5),
                    Point(0, 5),
                    Point(-3.5, 3.5),
                    Point(-5, 0),
                    Point(-3.5, -3.5),
                    Point(0, -5),
                    Point(3.5, -3.5),
                    Point(4, 1),
                    Point(0, 0),
                ],
            ),
            None,
            6.0,
            gpd.GeoDataFrame(
                {"id": [10], "priority": [100]},
                geometry=[Point(0, 0)],
            ),
        ),
    ],
    ids=[
        "no reference, no removals",
        "no reference, remove lower priority",
        "separate reference_gdf, no removal",
        "equal priority keeps candidate",
        "empty input_gdf",
        "chain should remove many points",
        "same point in both and removed",
        "circle with strong center point",
    ],
)
def test_reduce_nearby_points_selects_points_correctly(
    input_gdf: gpd.GeoDataFrame,
    reference_gdf: gpd.GeoDataFrame,
    distance_threshold: float,
    expected_gdf: gpd.GeoDataFrame,
):
    input_gdf = input_gdf.set_index("id")

    result_gdf = reduce_nearby_points_by_selecting(
        input_gdf,
        reference_gdf,
        distance_threshold,
        priority_column="priority",
    )

    expected_gdf = expected_gdf.set_index("id")

    result_sorted = result_gdf.sort_values("id").reset_index(drop=True)
    expected_sorted = expected_gdf.sort_values("id").reset_index(drop=True)

    # Check geometries
    for res_geom, exp_geom in zip(
        result_sorted.geometry, expected_sorted.geometry, strict=False
    ):
        assert res_geom.equals(exp_geom), f"Geometry mismatch: {res_geom} vs {exp_geom}"

    # Check attributes
    assert_frame_equal(
        result_sorted.drop(columns="geometry"),
        expected_sorted.drop(columns="geometry"),
        check_dtype=False,
    )


def test_reduce_nearby_points_by_selecting_raises_on_invalid_geometry():
    invalid_gdf = gpd.GeoDataFrame(
        {"id": [1], "priority": [10]},
        geometry=[LineString([(0, 0), (1, 1)])],
    )

    with pytest.raises(
        GeometryTypeError,
        match=re.escape(
            "reduce_nearby_points_by_selecting only supports Point geometries."
        ),
    ):
        reduce_nearby_points_by_selecting(invalid_gdf, None, 2.0, "priority")


def test_remove_small_polygons_a():  # test A - simple case of a few different-sized squares
    gdf = gpd.GeoDataFrame(
        {"id": [0, 1, 2]},
        geometry=[
            Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),  # 3 m x 3 m = 9 m²
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),  # 4 m x 4 m = 16 m²
            Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),  # 5 m x 5 m = 25 m²
        ],
        crs="EPSG:3067",
    )

    threshold = 12.0
    result = remove_small_polygons(gdf, area_threshold=threshold)

    for idx, _geom in enumerate(gdf.geometry):
        "remains" if idx in result["id"].to_numpy() else "is removed"

    assert len(result) == 2  # number of polygons that should remain


def test_remove_small_polygons_b():  # test B - handling None and empty geometries
    gdf = gpd.GeoDataFrame(
        {"id": [0, 1, 2]},
        geometry=[
            None,  # should be ignored
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon(),  # should be ignored
        ],
        crs="EPSG:3067",
    )

    threshold = 5.0
    result = remove_small_polygons(gdf, area_threshold=threshold)

    for idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            pass
        else:
            "remains" if idx in result["id"].to_numpy() else " is removed"

    assert len(result) == 0  # number of polygons that should remain


def test_remove_small_polygons_c():  # test C- handling CRS problems
    gdf = gpd.GeoDataFrame(
        {"id": [0]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",  # not projected (degrees)
    )

    with pytest.raises(ValueError, match="projected CRS"):
        remove_small_polygons(gdf, area_threshold=1.0)
