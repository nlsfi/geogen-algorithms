#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd
from pandas.testing import assert_frame_equal
from shapely import Point, equals_exact

from geogenalg.application.generalize_points import GeneralizePoints


def test_generalize_points() -> None:
    """
    Test generalize points algorithm.
    """

    algorithm = GeneralizePoints(
        reduce_threshold=0.5,
        displace_threshold=3,
        displace_points_iterations=10,
        unique_key_column="id",
        cluster_members_column="cluster_members",
    )

    input_gdf = gpd.GeoDataFrame(
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
    )

    result_gdf = algorithm.execute(input_gdf)

    expected_gdf = gpd.GeoDataFrame(
        {
            "id": [1, 3, 4, 6, 5, 7, 9, 10],
            "category": ["A", "C", "A", "B", "A", "B", "A", "A"],
            "cluster_members": [
                [1, 2],
                None,
                None,
                None,
                None,
                [7, 8],
                None,
                None,
            ],
        },
        geometry=[
            Point(0.15222, 0.04288, 1),
            Point(1.67094, 2.62775, 1),
            Point(-0.93985, 4.3516, 5),
            Point(5, 4, 0),
            Point(1.77584, 5.62341, 0),
            Point(3.27958, 8.21452, 0),
            Point(6.15843, 7.38223, 0),
            Point(9.15285, 7.50761, 0),
        ],
    )

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
