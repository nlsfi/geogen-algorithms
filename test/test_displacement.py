#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
import re

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal
from shapely import equals_exact
from shapely.geometry import LineString, Point, Polygon

from geogenalg import displacement
from geogenalg.core.exceptions import GeometryTypeError


@pytest.mark.parametrize(
    ("input_gdf", "threshold", "iterations", "expected_gdf"),
    [
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0, 2)]),
            1.0,
            1,
            gpd.GeoDataFrame(geometry=[Point(0, 0, 2)]),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(2, 0)]),
            1.0,
            1,
            gpd.GeoDataFrame(geometry=[Point(0, 0, 0), Point(2, 0, 0)]),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0.5, 0)]),
            1.0,
            1,
            gpd.GeoDataFrame(geometry=[Point(-0.25, 0, 0), Point(0.75, 0, 0)]),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0, 1), Point(1, 0, 1), Point(2, 0, 1)]),
            2.0,
            10,
            gpd.GeoDataFrame(
                geometry=[Point(-1, 0, 1), Point(1, 0, 1), Point(3, 0, 1)]
            ),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0.5, 0), Point(5, 5)]),
            1.0,
            5,
            gpd.GeoDataFrame(
                geometry=[Point(-0.25, 0, 0), Point(0.75, 0, 0), Point(5, 5, 0)]
            ),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0, 1), Point(0, 0, 2), Point(0, 0, 3)]),
            1.0,
            2,
            gpd.GeoDataFrame(geometry=[Point(0, 0, 1), Point(0, 0, 2), Point(0, 0, 3)]),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1.0, 0)]),
            1.0,
            1,
            gpd.GeoDataFrame(geometry=[Point(0, 0, 0), Point(1.0, 0, 0)]),
        ),
    ],
    ids=[
        "single point should remain unmodified",
        "two far apart should remain unmodified",
        "two close should be displaced",
        "three close points should displace outer points",
        "two close and one far should displace the close pair",
        "three points in the same location should remain unmodified",
        "distance exactly equal to threshold should not displace",
    ],
)
def test_displace_points_moves_points_as_expected(
    input_gdf: gpd.GeoDataFrame,
    threshold: float,
    iterations: int,
    expected_gdf: gpd.GeoDataFrame,
):
    result_gdf = displacement.displace_points(input_gdf, threshold, iterations)

    for _, (result_geometry, expected_geometry) in enumerate(
        zip(
            result_gdf.geometry.sort_index().reset_index(drop=True),
            expected_gdf.geometry.sort_index().reset_index(drop=True),
            strict=False,
        )
    ):
        assert equals_exact(result_geometry, expected_geometry, tolerance=1e-3)

        # Shapely 2.0 equals_exact compares only XY coordinates
        if result_geometry.has_z or expected_geometry.has_z:
            assert abs(result_geometry.z - expected_geometry.z) < 1e-3


def test_displace_points_preserves_attributes_intact():
    input_gdf = gpd.GeoDataFrame(
        {"id": [5, 4, 6], "category": ["Z", "X", "Y"]},
        geometry=[Point(0, 0), Point(0.5, 0), Point(5, 5)],
    )

    expected_gdf = gpd.GeoDataFrame(
        {"id": [5, 4, 6], "category": ["Z", "X", "Y"]},
        geometry=[Point(-0.25, 0), Point(0.75, 0), Point(5, 5)],
    )

    result_gdf = displacement.displace_points(
        input_gdf, displace_threshold=1.0, iterations=4
    )

    result_attrs = result_gdf.drop(columns="geometry").reset_index(drop=True)
    expected_attrs = expected_gdf.drop(columns="geometry").reset_index(drop=True)
    assert_frame_equal(result_attrs, expected_attrs)


def test_displace_points_raises_on_non_point_geometry():
    invalid_input_gdf = gpd.GeoDataFrame(
        geometry=[
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ]
    )
    displace_threshold = 1.0
    iterations = 5

    with pytest.raises(
        GeometryTypeError,
        match=re.escape("displace_points only supports Point geometries."),
    ):
        displacement.displace_points(invalid_input_gdf, displace_threshold, iterations)
