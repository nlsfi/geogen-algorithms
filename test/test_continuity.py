#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from collections import defaultdict

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import LineString, MultiLineString

from geogenalg.continuity import (
    check_line_connections,
    check_reference_line_connections,
    connect_nearby_endpoints,
    detect_dead_ends,
    find_all_endpoints,
    flag_connections,
    flag_connections_to_reference,
    get_paths_along_roads,
    inspect_dead_end_candidates,
)


@pytest.mark.parametrize(
    ("lines", "expected_results"),
    [
        (
            [LineString([(0, 0), (1, 1)])],
            [((0, 0), 1), ((1, 1), 1)],
        ),
        (
            [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
            [((0, 0), 1), ((1, 1), 2), ((2, 2), 1)],
        ),
        (
            [MultiLineString([[(0, 0), (0.5, 0.5), (1, 0)], [(1, 0), (2, 0)]])],
            [((0, 0), 1), ((1, 0), 2), ((2, 0), 1)],
        ),
        (
            [LineString([])],
            [],
        ),
    ],
    ids=["single_line", "shared_endpoint", "multi_line", "empty_line"],
)
def test_find_all_endpoints(lines: list, expected_results: list):
    result = find_all_endpoints(lines)
    coord_count = defaultdict(int)
    for pt, _, count in result:
        key = (round(pt.x, 5), round(pt.y, 5))
        coord_count[key] = count

    expected_dict = dict(expected_results)

    assert len(coord_count) == len(expected_dict)
    for coord, count in expected_dict.items():
        assert coord in coord_count
        assert coord_count[coord] == count


@pytest.mark.parametrize(
    ("input_gdf", "gap_threshold", "expected_helper_lines"),
    [
        (
            GeoDataFrame(
                geometry=[LineString([(0, 0), (1, 0)]), LineString([(1.1, 0), (2, 0)])],
                crs="EPSG:3067",
            ),
            0.2,
            [LineString([(1, 0), (1.1, 0)]), LineString([(1.1, 0), (1, 0)])],
        ),
        (
            GeoDataFrame(
                geometry=[LineString([(0, 0), (1, 0)]), LineString([(2, 0), (3, 0)])],
                crs="EPSG:3067",
            ),
            0.5,
            [],
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0.1), (0, 2)]),
                    LineString([(-0.1, 0.1), (-3, 3)]),
                ],
                crs="EPSG:3067",
            ),
            0.2,
            [
                LineString([(1, 0), (1, 0.1)]),
                LineString([(0, 0), (-0.1, 0.1)]),
                LineString([(1, 0.1), (1, 0)]),
                LineString([(-0.1, 0.1), (0, 0)]),
            ],
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0, 0), (1, 0, 0)]),
                    LineString([(1.1, 0, 0), (2, 0, 0)]),
                ],
                crs="EPSG:3903",
            ),
            0.2,
            [
                LineString([(1, 0, 0), (1.1, 0, 0)]),
                LineString([(1.1, 0, 0), (1, 0, 0)]),
            ],
        ),
    ],
    ids=[
        "within_threshold",
        "outside_threshold",
        "multiple_connections",
        "coordinates_with_height",
    ],
)
def test_connect_nearby_endpoints_when_gap_within_threshold(
    input_gdf: GeoDataFrame,
    gap_threshold: float,
    expected_helper_lines: list[LineString],
):
    result = connect_nearby_endpoints(input_gdf, gap_threshold)
    assert isinstance(result, GeoDataFrame)
    assert len(result) == len(expected_helper_lines)
    for geom in result.geometry:
        assert geom.length <= gap_threshold
        assert geom in expected_helper_lines


@pytest.mark.parametrize(
    (
        "input_gdf",
        "threshold_distance",
        "expected_number_of_connected",
        "expected_number_of_unconnected",
    ),
    [
        (
            GeoDataFrame(
                geometry=[LineString([(0, 0), (1, 0)]), LineString([(1.5, 0), (5, 0)])],
                crs="EPSG:3067",
            ),
            1.0,
            2,
            0,
        ),
        (
            GeoDataFrame(
                geometry=[LineString([(0, 0), (1, 0)]), LineString([(5, 0), (10, 0)])],
                crs="EPSG:3067",
            ),
            2.0,
            0,
            2,
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (1, 0)]),
                    LineString([(2, 0), (5, 0)]),
                    LineString([(0, 2), (0, 5)]),
                ],
                crs="EPSG:3067",
            ),
            1.5,
            2,
            1,
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0, 0), (1, 0, 0)]),
                    LineString([(1.1, 0, 0), (2, 0, 0)]),
                ],
                crs="EPSG:3903",
            ),
            0.2,
            2,
            0,
        ),
    ],
    ids=[
        "within_threshold",
        "outside_threshold",
        "multiple_connections",
        "coordinates_with_height",
    ],
)
def test_check_line_connections(
    input_gdf: GeoDataFrame,
    threshold_distance: float,
    expected_number_of_connected: int,
    expected_number_of_unconnected: int,
):
    result = check_line_connections(input_gdf, threshold_distance)
    assert isinstance(result[0], GeoDataFrame)
    assert isinstance(result[1], GeoDataFrame)
    assert len(result[0].index) == expected_number_of_connected
    assert len(result[1].index) == expected_number_of_unconnected


@pytest.mark.parametrize(
    (
        "input_gdf",
        "threshold_distance",
        "reference_gdf_list",
        "expected_number_of_connected",
        "expected_number_of_unconnected",
    ),
    [
        (
            GeoDataFrame(
                geometry=[LineString([(0, 0), (1, 0)])],
                crs="EPSG:3067",
            ),
            0.1,
            [
                GeoDataFrame(
                    geometry=[LineString([(1, 0), (2, 0)])],
                    crs="EPSG:3067",
                )
            ],
            1,
            0,
        ),
        (
            GeoDataFrame(
                geometry=[LineString([(0, 0), (1, 0)])],
                crs="EPSG:3067",
            ),
            2.0,
            [
                GeoDataFrame(
                    geometry=[LineString([(5, 0), (10, 0)])],
                    crs="EPSG:3067",
                )
            ],
            0,
            1,
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0, 0), (1, 0, 0)]),
                    LineString([(2, 0, 0), (3, 0, 0)]),
                ],
                crs="EPSG:3903",
            ),
            0.2,
            [
                GeoDataFrame(
                    geometry=[
                        LineString([(0, 0, 0), (0, 1, 0)]),
                    ],
                    crs="EPSG:3903",
                )
            ],
            1,
            1,
        ),
    ],
    ids=[
        "within_threshold",
        "outside_threshold",
        "coordinates_with_height",
    ],
)
def test_check_reference_line_connections(
    input_gdf: GeoDataFrame,
    threshold_distance: float,
    reference_gdf_list: list[GeoDataFrame],
    expected_number_of_connected: int,
    expected_number_of_unconnected: int,
):
    result = check_reference_line_connections(
        input_gdf, threshold_distance, reference_gdf_list
    )
    assert isinstance(result[0], GeoDataFrame)
    assert isinstance(result[1], GeoDataFrame)
    assert len(result[0].index) == expected_number_of_connected
    assert len(result[1].index) == expected_number_of_unconnected
    assert "is_connected" in result[0].columns


def test_detect_dead_ends():
    gdf = GeoDataFrame(
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
            LineString([(10, 0), (11, 0)]),
        ],
        crs="EPSG:3067",
    )

    normal, dead_end = detect_dead_ends(gdf, threshold_distance=0.01)

    assert len(normal) == 1
    assert len(dead_end) == 2
    assert "first_intersects" in dead_end.columns
    assert "last_intersects" in dead_end.columns


def test_inspect_dead_end_candidates():
    gdf = GeoDataFrame(
        {
            "first_intersects": [True, False],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(2, 0), (3, 0)]),
            ],
        },
        crs="EPSG:3067",
    )

    reference = GeoDataFrame(
        geometry=[LineString([(1, 0), (2, 0)])],
        crs="EPSG:3067",
    )

    result = inspect_dead_end_candidates(
        gdf,
        threshold_distance=0.1,
        reference_gdf_list=[reference],
    )

    assert "dead_end_connects_to_ref_gdf" in result.columns
    assert result["dead_end_connects_to_ref_gdf"].any()


@pytest.mark.parametrize(
    ("gdf_input", "gdf_reference", "detection_distance", "expected_number_along_roads"),
    [
        (
            GeoDataFrame(
                geometry=[LineString([(0, 0), (1, 0)]), LineString([(3, 0), (5, 0)])],
                crs="EPSG:3067",
            ),
            GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)])], crs="EPSG:3067"),
            1.0,
            2,
        ),
        (
            GeoDataFrame(
                geometry=[LineString([(0, 0), (1, 0)]), LineString([(2, 0), (3, 0)])],
                crs="EPSG:3067",
            ),
            GeoDataFrame(
                geometry=[LineString([(0, 0), (1, 0)])],
                crs="EPSG:3067",
            ),
            1.0,
            1,
        ),
    ],
    ids=[
        "both_inside_buffer",
        "one_inside_buffer",
    ],
)
def test_get_paths_along_roads(
    gdf_input: GeoDataFrame,
    gdf_reference: GeoDataFrame,
    detection_distance: float,
    expected_number_along_roads: int,
):
    result = get_paths_along_roads(gdf_input, gdf_reference, detection_distance)
    assert isinstance(result[0], GeoDataFrame)
    assert isinstance(result[1], GeoDataFrame)
    assert len(result[0].index) == expected_number_along_roads


@pytest.mark.parametrize(
    (
        "input_gdf",
        "expected_gdf",
    ),
    [
        (
            GeoDataFrame(
                geometry=[
                    LineString([(5, 1), (5, 0)]),
                    LineString([(5, 1), (5, 10)]),
                ],
            ),
            GeoDataFrame(
                {
                    "__start_connected": [
                        True,
                        True,
                    ],
                    "__end_connected": [
                        False,
                        False,
                    ],
                },
                geometry=[
                    LineString([(5, 1), (5, 0)]),
                    LineString([(5, 1), (5, 10)]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString([(5, 1), (5, 0)]),
                    LineString([(0, 10), (5, 10)]),
                ],
            ),
            GeoDataFrame(
                {
                    "__start_connected": [
                        False,
                        False,
                    ],
                    "__end_connected": [
                        False,
                        False,
                    ],
                },
                geometry=[
                    LineString([(5, 1), (5, 0)]),
                    LineString([(0, 10), (5, 10)]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (1, 0)]),
                    LineString([(0, 1), (1, 1)]),
                    LineString([(0, 0), (0, 1)]),
                ],
            ),
            GeoDataFrame(
                {
                    "__start_connected": [
                        True,
                        True,
                        True,
                    ],
                    "__end_connected": [
                        False,
                        False,
                        True,
                    ],
                },
                geometry=[
                    LineString([(0, 0), (1, 0)]),
                    LineString([(0, 1), (1, 1)]),
                    LineString([(0, 0), (0, 1)]),
                ],
            ),
        ),
    ],
    ids=[
        "connected_at_start",
        "no_connections",
        "both_ends_connected_for_one",
    ],
)
def test_flag_connections(
    input_gdf: GeoDataFrame,
    expected_gdf: GeoDataFrame,
):
    assert_geodataframe_equal(
        flag_connections(input_gdf),
        expected_gdf,
        check_like=True,
    )


@pytest.mark.parametrize(
    (
        "input_gdf",
        "reference_gdf",
        "expected_gdf",
    ),
    [
        (
            GeoDataFrame(
                geometry=[
                    LineString([(5, 1), (5, 0)]),
                    LineString([(7, 0), (7, 1)]),
                ],
            ),
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (10, 0)]),
                ],
            ),
            GeoDataFrame(
                {
                    "__start_connected": [
                        False,
                        True,
                    ],
                    "__end_connected": [
                        True,
                        False,
                    ],
                },
                geometry=[
                    LineString([(5, 1), (5, 0)]),
                    LineString([(7, 0), (7, 1)]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString([(5, 1), (5, 2)]),
                ],
            ),
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (10, 0)]),
                ],
            ),
            GeoDataFrame(
                {
                    "__start_connected": [
                        False,
                    ],
                    "__end_connected": [
                        False,
                    ],
                },
                geometry=[
                    LineString([(5, 1), (5, 2)]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString([(5, 2), (5, 0)]),
                ],
            ),
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (10, 0)]),
                    LineString([(0, 2), (10, 2)]),
                ],
            ),
            GeoDataFrame(
                {
                    "__start_connected": [
                        True,
                    ],
                    "__end_connected": [
                        True,
                    ],
                },
                geometry=[
                    LineString([(5, 2), (5, 0)]),
                ],
            ),
        ),
    ],
    ids=[
        "one connected at start, one at end",
        "no connections",
        "both connected",
    ],
)
def test_flag_connections_to_reference(
    input_gdf: GeoDataFrame,
    reference_gdf: GeoDataFrame,
    expected_gdf: GeoDataFrame,
):
    assert_geodataframe_equal(
        flag_connections_to_reference(input_gdf, reference_gdf),
        expected_gdf,
        check_like=True,
    )
