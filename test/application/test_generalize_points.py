#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
from geopandas import GeoDataFrame
from pandas.testing import assert_frame_equal
from shapely import LineString, Point, equals_exact

from geogenalg.application.generalize_points import GeneralizePoints
from geogenalg.core.exceptions import GeometryTypeError


def test_generalize_points() -> None:
    """
    Test generalize points algorithm.
    """

    algorithm = GeneralizePoints(
        cluster_distance=1.0,
        displace_threshold=3,
        displace_points_iterations=10,
    )

    input_gdf = GeoDataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "category": ["A", "A", "C", "A", "A", "B", "B", "B", "A", "A"],
        },
        geometry=[
            Point(0, 0, 1),
            Point(0.5, 0.5),
            Point(1, 3, 1),
            Point(0, 4, 5),
            Point(2, 6),
            Point(5, 4),
            Point(4, 7.1),
            Point(4, 7.9),
            Point(6, 7.5),
            Point(8, 7.5),
        ],
    ).set_index("id")

    result_gdf = algorithm.execute(input_gdf, {})

    expected_gdf = GeoDataFrame(
        {
            "id": [
                "d0b31c3c01113c9ecdc7cbc2cb1934c2d22fbd6df67b8c55951fa9edf436c18c",
                "5c96371515d2a8caf840945a3b4b83d1340ca42705d6768f0ad54bfb28370029",
                "3",
                "4",
                "5",
                "6",
                "9",
                "10",
            ],
            "category": ["A", "B", "C", "A", "A", "B", "A", "A"],
            "feature_type": [
                "centroid_from_point",
                "centroid_from_point",
                "unchanged_point",
                "unchanged_point",
                "unchanged_point",
                "unchanged_point",
                "unchanged_point",
                "unchanged_point",
            ],
        },
        geometry=[
            Point(0.15222, 0.04288, 1),
            Point(3.27958, 8.21452, 0),
            Point(1.67094, 2.62775, 1),
            Point(-0.93985, 4.3516, 5),
            Point(1.77584, 5.62341, 0),
            Point(5, 4, 0),
            Point(6.15843, 7.38223, 0),
            Point(9.15285, 7.50761, 0),
        ],
    )
    expected_gdf = expected_gdf.set_index("id")

    # Check geometries
    for _, (result_geometry, expected_geometry) in enumerate(
        zip(
            result_gdf.geometry.reset_index(drop=True),
            expected_gdf.geometry.reset_index(drop=True),
            strict=False,
        )
    ):
        assert equals_exact(result_geometry, expected_geometry, tolerance=1e-3)

        # Shapely 2.0 equals_exact compares only XY coordinates
        if result_geometry.has_z or expected_geometry.has_z:
            assert abs(result_geometry.z - expected_geometry.z) < 1e-3

    # Check attributes
    result_attrs = result_gdf.drop(columns="geometry").reset_index(drop=True)
    expected_attrs = expected_gdf.drop(columns="geometry").reset_index(drop=True)
    assert_frame_equal(result_attrs, expected_attrs)

    # Check IDs
    assert len(result_gdf.index.difference(expected_gdf.index)) == 0


def test_generalize_points_invalid_geometry_type():
    with pytest.raises(
        GeometryTypeError,
        match=r"Input data must contain only geometries of following types: Point.",
    ):
        GeneralizePoints().execute(
            data=GeoDataFrame(
                {}, geometry=[LineString([Point(0.5, 0.5), Point(1.0, 1.0)])]
            ),
            reference_data={},
        )
