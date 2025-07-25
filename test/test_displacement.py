#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal
from shapely import equals_exact
from shapely.geometry import Point

from geogenalg import displacement


@pytest.mark.parametrize(
    ("input_gdf", "threshold", "expected_gdf"),
    [
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0)]),
            1.0,
            gpd.GeoDataFrame(geometry=[Point(0, 0)]),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(2, 0)]),
            1.0,
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(2, 0)]),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0.5, 0)]),
            1.0,
            gpd.GeoDataFrame(geometry=[Point(-0.25, 0), Point(0.75, 0)]),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 0), Point(2, 0)]),
            2.0,
            gpd.GeoDataFrame(geometry=[Point(-0.5, 0), Point(1, 0), Point(2.5, 0)]),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0.5, 0), Point(5, 5)]),
            1.0,
            gpd.GeoDataFrame(geometry=[Point(-0.25, 0), Point(0.75, 0), Point(5, 5)]),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0, 0), Point(0, 0)]),
            1.0,
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0, 0), Point(0, 0)]),
        ),
        (
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1.0, 0)]),
            1.0,
            gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1.0, 0)]),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [5, 4, 6], "category": ["Z", "X", "Y"]},
                geometry=[Point(0, 0), Point(0.5, 0), Point(5, 5)],
            ),
            1.0,
            gpd.GeoDataFrame(
                {"id": [5, 4, 6], "category": ["Z", "X", "Y"]},
                geometry=[Point(-0.25, 0), Point(0.75, 0), Point(5, 5)],
            ),
        ),
    ],
    ids=[
        "single_point",
        "two_far_apart",
        "two_close_displaced",
        "three_close_displaced",
        "two_close_one_far",
        "three_points_same_location",
        "distance_exactly_threshold",
        "three_points_with_attributes",
    ],
)
def test_displace_points(
    input_gdf: gpd.GeoDataFrame, threshold: float, expected_gdf: gpd.GeoDataFrame
):
    result_gdf = displacement.displace_points(input_gdf, threshold)

    # Check geometries
    for _i, (result_geometry, expected_geometry) in enumerate(
        zip(
            result_gdf.geometry.sort_index().reset_index(drop=True),
            expected_gdf.geometry.sort_index().reset_index(drop=True),
            strict=False,
        )
    ):
        assert equals_exact(result_geometry, expected_geometry, tolerance=1e-4)

    # Check attributes
    result_attrs = result_gdf.drop(columns="geometry").reset_index(drop=True)
    expected_attrs = expected_gdf.drop(columns="geometry").reset_index(drop=True)
    assert_frame_equal(result_attrs, expected_attrs)
