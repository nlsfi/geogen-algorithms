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
from shapely import equals_exact
from shapely.geometry import LineString, Point, Polygon

from geogenalg import cluster
from geogenalg.core.exceptions import GeometryTypeError


@pytest.mark.parametrize(
    ("input_gdf", "threshold", "unique_id_column", "expected_gdf"),
    [
        (
            gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0, 1)]),
            1.0,
            "id",
            gpd.GeoDataFrame(
                {"id": [1], "cluster_members": [None]}, geometry=[Point(0, 0, 1)]
            ),
        ),
        (
            gpd.GeoDataFrame({"id": [2, 1]}, geometry=[Point(0, 0), Point(0.5, 0)]),
            1.0,
            "id",
            gpd.GeoDataFrame(
                {"id": [1], "cluster_members": [[2, 1]]}, geometry=[Point(0.25, 0, 0.0)]
            ),
        ),
        (
            gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(0, 0, 1), Point(5, 5, 2)]),
            1.0,
            "id",
            gpd.GeoDataFrame(
                {"id": [1, 2], "cluster_members": [None, None]},
                geometry=[Point(0, 0, 1), Point(5, 5, 2)],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [5, 4, 6]},
                geometry=[Point(0, 0, 0), Point(0.5, 0, 1), Point(5, 5)],
            ),
            1.0,
            "id",
            gpd.GeoDataFrame(
                {"id": [4, 6], "cluster_members": [[5, 4], None]},
                geometry=[Point(0.25, 0, 0.5), Point(5, 5, 0.0)],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [10, 9, 8, 7]},
                geometry=[Point(0, 0), Point(1, 0, 2), Point(0, 1, 3), Point(1, 1)],
            ),
            2.0,
            "id",
            gpd.GeoDataFrame(
                {"id": [7], "cluster_members": [[10, 9, 8, 7]]},
                geometry=[Point(0.5, 0.5, 2.5)],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [3, 2, 1]}, geometry=[Point(1, 1), Point(1, 1), Point(1, 1)]
            ),
            0.5,
            "id",
            gpd.GeoDataFrame(
                {"id": [1], "cluster_members": [[3, 2, 1]]},
                geometry=[Point(1, 1, 0.0)],
            ),
        ),
        (
            gpd.GeoDataFrame({"id": [1], "name": ["A"]}, geometry=[Point(0, 0)]),
            1.0,
            "id",
            gpd.GeoDataFrame(
                {"id": [1], "name": ["A"], "cluster_members": [None]},
                geometry=[Point(0, 0, 0.0)],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [2, 1], "name": ["B", "A"]},
                geometry=[Point(0, 0), Point(0.5, 0)],
            ),
            1.0,
            "id",
            gpd.GeoDataFrame(
                {
                    "id": [1],
                    "name": ["A"],
                    "cluster_members": [[2, 1]],
                },
                geometry=[Point(0.25, 0, 0.0)],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [5, 4, 6], "category": ["Z", "X", "Y"]},
                geometry=[Point(0, 0), Point(0.5, 0), Point(5, 5)],
            ),
            1.0,
            "id",
            gpd.GeoDataFrame(
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
    input_gdf: gpd.GeoDataFrame,
    threshold: float,
    unique_id_column: str,
    expected_gdf: gpd.GeoDataFrame,
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
    invalid_input_gdf = gpd.GeoDataFrame(
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
