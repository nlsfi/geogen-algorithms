#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import re
from collections.abc import Callable

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from pandas import Series
from pandas.testing import assert_frame_equal, assert_series_equal
from shapely import equals_exact
from shapely.geometry import LineString, Point, Polygon

from geogenalg import cluster
from geogenalg.core.exceptions import GeometryTypeError


@pytest.mark.parametrize(
    ("input_gdf", "threshold", "unique_id_column", "expected_gdf"),
    [
        (
            GeoDataFrame({"id": [1]}, geometry=[Point(0, 0, 1)]),
            1.0,
            "id",
            GeoDataFrame(
                {"id": [1], "cluster_members": [None]}, geometry=[Point(0, 0, 1)]
            ),
        ),
        (
            GeoDataFrame({"id": [2, 1]}, geometry=[Point(0, 0), Point(0.5, 0)]),
            1.0,
            "id",
            GeoDataFrame(
                {"id": [1], "cluster_members": [[2, 1]]}, geometry=[Point(0.25, 0, 0.0)]
            ),
        ),
        (
            GeoDataFrame({"id": [1, 2]}, geometry=[Point(0, 0, 1), Point(5, 5, 2)]),
            1.0,
            "id",
            GeoDataFrame(
                {"id": [1, 2], "cluster_members": [None, None]},
                geometry=[Point(0, 0, 1), Point(5, 5, 2)],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [5, 4, 6]},
                geometry=[Point(0, 0, 0), Point(0.5, 0, 1), Point(5, 5)],
            ),
            1.0,
            "id",
            GeoDataFrame(
                {"id": [4, 6], "cluster_members": [[5, 4], None]},
                geometry=[Point(0.25, 0, 0.5), Point(5, 5, 0.0)],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [10, 9, 8, 7]},
                geometry=[Point(0, 0), Point(1, 0, 2), Point(0, 1, 3), Point(1, 1)],
            ),
            2.0,
            "id",
            GeoDataFrame(
                {"id": [7], "cluster_members": [[10, 9, 8, 7]]},
                geometry=[Point(0.5, 0.5, 2.5)],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [3, 2, 1]}, geometry=[Point(1, 1), Point(1, 1), Point(1, 1)]
            ),
            0.5,
            "id",
            GeoDataFrame(
                {"id": [1], "cluster_members": [[3, 2, 1]]},
                geometry=[Point(1, 1, 0.0)],
            ),
        ),
        (
            GeoDataFrame({"id": [1], "name": ["A"]}, geometry=[Point(0, 0)]),
            1.0,
            "id",
            GeoDataFrame(
                {"id": [1], "name": ["A"], "cluster_members": [None]},
                geometry=[Point(0, 0, 0.0)],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [2, 1], "name": ["B", "A"]},
                geometry=[Point(0, 0), Point(0.5, 0)],
            ),
            1.0,
            "id",
            GeoDataFrame(
                {
                    "id": [1],
                    "name": ["A"],
                    "cluster_members": [[2, 1]],
                },
                geometry=[Point(0.25, 0, 0.0)],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [5, 4, 6], "category": ["Z", "X", "Y"]},
                geometry=[Point(0, 0), Point(0.5, 0), Point(5, 5)],
            ),
            1.0,
            "id",
            GeoDataFrame(
                {
                    "id": [4, 6],
                    "category": ["X", "Y"],
                    "cluster_members": [[5, 4], None],
                },
                geometry=[Point(0.25, 0, 0.0), Point(5, 5, 0.0)],
            ),
        ),
    ],
    ids=[
        "single point should remain unmodified",
        "two close should be clustered",
        "two far apart should remain separate",
        "two close and one far should cluster only the close pair",
        "four should be clustered together",
        "three points at same location should cluster",
        "single point with attributes",
        "two points with attributes",
        "three points with attributes",
    ],
)
def test_reduce_nearby_points_by_clustering_correctly(
    input_gdf: GeoDataFrame,
    threshold: float,
    unique_id_column: str,
    expected_gdf: GeoDataFrame,
):
    result_gdf = cluster.reduce_nearby_points_by_clustering(
        input_gdf, threshold, unique_id_column
    )

    # Check geometries
    for _, (result_geometry, expected_geometry) in enumerate(
        zip(
            result_gdf.geometry.sort_index().reset_index(drop=True),
            expected_gdf.geometry.sort_index().reset_index(drop=True),
            strict=False,
        )
    ):
        assert equals_exact(result_geometry, expected_geometry, tolerance=1e-5)

        # Shapely 2.0 equals_exact compares only XY coordinates
        if result_geometry.has_z or expected_geometry.has_z:
            assert abs(result_geometry.z - expected_geometry.z) < 1e-5

    # Check attributes
    result_attrs = (
        result_gdf.drop(columns="geometry").sort_values("id").reset_index(drop=True)
    )
    expected_attrs = (
        expected_gdf.drop(columns="geometry").sort_values("id").reset_index(drop=True)
    )
    assert_frame_equal(result_attrs, expected_attrs)


def test_reduce_nearby_points_raises_on_non_point_geometry():
    invalid_input_gdf = GeoDataFrame(
        geometry=[
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ]
    )
    threshold = 1.0
    unique_id_column = "id"

    with pytest.raises(
        GeometryTypeError,
        match=re.escape("reduce_nearby_points only supports Point geometries."),
    ):
        cluster.reduce_nearby_points_by_clustering(
            invalid_input_gdf, threshold, unique_id_column
        )


@pytest.mark.parametrize(
    (
        "input_gdf",
        "cluster_distance",
        "cluster_column_name",
        "ignore_z",
        "expected_series",
    ),
    [
        (
            GeoDataFrame(geometry=[Point(0, 0), Point(0.5, 0), Point(2.0, 0)]),
            1.0,
            "cluster_id",
            True,
            Series([0, 0, -1], name="cluster_id"),
        ),
        (
            GeoDataFrame(geometry=[Point(0, 0), Point(0.5, 0), Point(0, 0.1)]),
            1.0,
            "cluster_id",
            True,
            Series([0, 0, 0], name="cluster_id"),
        ),
        (
            GeoDataFrame(
                geometry=[
                    Point(0, 0),
                    Point(0.5, 0),
                    Point(2.0, 2.0),
                    Point(2.1, 2.1),
                    Point(5.0, 5.0),
                ]
            ),
            1.0,
            "cluster_id",
            True,
            Series([0, 0, 1, 1, -1], name="cluster_id"),
        ),
        (
            GeoDataFrame(geometry=[Point(0, 0, 4), Point(0.5, 0, 0)]),
            1.0,
            "cluster_id",
            False,
            Series([-1, -1], name="cluster_id"),
        ),
        (
            GeoDataFrame(geometry=[Point(0, 0, 4), Point(0.5, 0, 0)]),
            1.0,
            "cluster_id",
            True,
            Series([0, 0], name="cluster_id"),
        ),
        (
            GeoDataFrame(geometry=[Point(0, 0)]),
            1.0,
            "cid",
            True,
            Series([-1], name="cid"),
        ),
    ],
    ids=[
        "three points, two of which clustered",
        "three points, all clustered",
        "five points, two clusters",
        "ignore z false",
        "ignore z true",
        "one point, different series name",
    ],
)
def test_dbscan_cluster(
    input_gdf: GeoDataFrame,
    cluster_distance: float,
    cluster_column_name: str,
    ignore_z: bool,
    expected_series: Series,
):
    result = cluster.dbscan_cluster_ids(
        input_gdf,
        cluster_distance,
        cluster_column_name=cluster_column_name,
        ignore_z=ignore_z,
    )

    assert_series_equal(result, expected_series)


def _aggregate_test_function_sum(data: Series) -> float:
    # define this function to test that a function can be passed into
    # get_cluster_centroids' aggregation functions
    return data.sum()


@pytest.mark.parametrize(
    (
        "input_gdf",
        "cluster_distance",
        "aggregation_functions",
        "expected_gdf",
    ),
    [
        (
            GeoDataFrame(
                {"id": [1, 2, 3]}, geometry=[Point(0, 0), Point(0.5, 0), Point(2.0, 0)]
            ),
            1.0,
            None,
            GeoDataFrame({"id": [1], "old_ids": [(1, 2)]}, geometry=[Point(0.25, 0)]),
        ),
        (
            GeoDataFrame(
                {"id": [1, 2, 3, 10]},
                geometry=[Point(0, 1), Point(0.5, 1), Point(2.0, 2), Point(4.0, 2)],
            ),
            5.0,
            None,
            GeoDataFrame(
                {"id": [1], "old_ids": [(1, 2, 3, 10)]}, geometry=[Point(1.625, 1.5)]
            ),
        ),
        (
            GeoDataFrame(
                {"id": [1, 2, 3, 4, 5]},
                geometry=[
                    Point(0, 1),
                    Point(0.5, 1),
                    Point(3.5, 2),
                    Point(4.0, 2),
                    Point(40.0, 20.0),
                ],
            ),
            1.0,
            None,
            GeoDataFrame(
                {"id": [1, 3], "old_ids": [(1, 2), (3, 4)]},
                geometry=[Point(0.25, 1), Point(3.75, 2)],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2],
                    "numeric_column_mean": [40.0, 60.0],
                    "numeric_column_sum": [40.0, 60.0],
                    "str_column": ["string_1", "string_2"],
                },
                geometry=[Point(0, 1), Point(0.5, 1)],
            ),
            1.0,
            {
                "id": "last",
                "numeric_column_mean": "mean",
                "numeric_column_sum": _aggregate_test_function_sum,
                "str_column": ", ".join,
            },
            GeoDataFrame(
                {
                    "id": [2],
                    "numeric_column_mean": [50.0],
                    "numeric_column_sum": [100.0],
                    "str_column": ["string_1, string_2"],
                    "old_ids": [(1, 2)],
                },
                geometry=[Point(0.25, 1)],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "numeric_column": [1.0, 2.0, 100.0, 200.0],
                },
                geometry=[
                    Point(0, 1),
                    Point(0.5, 1),
                    Point(100.0, 100.0),
                    Point(101.0, 100.0),
                ],
            ),
            1.0,
            {
                "numeric_column": "mean",
            },
            GeoDataFrame(
                {
                    "id": [1, 3],
                    "numeric_column": [1.5, 150.0],
                    "old_ids": [(1, 2), (3, 4)],
                },
                geometry=[Point(0.25, 1), Point(100.5, 100.0)],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [1, 2]}, geometry=[Point(0.0, 1.0, 0.0), Point(0.5, 1.0, 100.0)]
            ),
            1.0,
            None,
            GeoDataFrame(
                {"id": [1], "old_ids": [(1, 2)]}, geometry=[Point(0.25, 1, 50.0)]
            ),
        ),
    ],
    ids=[
        "three points, two clustered",
        "four points, all clustered",
        "five points, two clusters",
        "aggregation functions",
        "aggregation functions, two clusters",
        "geometries with z",
    ],
)
def test_get_cluster_centroids(
    input_gdf: GeoDataFrame,
    cluster_distance: float,
    aggregation_functions: dict[str, Callable | str] | None,
    expected_gdf: GeoDataFrame,
):
    input_gdf = input_gdf.set_index(input_gdf["id"])

    result = cluster.get_cluster_centroids(
        input_gdf,
        cluster_distance,
        aggregation_functions=aggregation_functions,
    )

    assert_geodataframe_equal(result, expected_gdf, check_like=True)
