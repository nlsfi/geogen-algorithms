#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path

import geopandas as gpd
import geopandas.testing
from pandas.testing import assert_frame_equal
from shapely import Point, equals_exact

from geogenalg.application.generalize_points import (
    AlgorithmOptions,
    generalize_points,
)


def test_generalize_points(
    testdata_path: Path,
) -> None:
    """
    Test generalize points algorithm
    """

    options = AlgorithmOptions(
        reduce_threshold=0.5,
        displace_threshold=3,
        iterations=10,
        unique_key_column="id",
        cluster_members_column="cluster_members",
    )

    input_gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "category": ["A", "A", "C", "A", "A", "B", "B", "B", "A", "A"],
        },
        geometry=[
            Point(0, 0),
            Point(0.5, 0.5),
            Point(1, 3),
            Point(0, 4),
            Point(2, 6),
            Point(5, 4),
            Point(4, 7.1),
            Point(4, 7.9),
            Point(6, 7.5),
            Point(8, 7.5),
        ],
    )

    result_gdf = generalize_points(input_gdf, options)

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
            Point(0.15222, 0.04288),
            Point(1.67094, 2.62775),
            Point(-0.93985, 4.3516),
            Point(5, 4),
            Point(1.77584, 5.62341),
            Point(3.27958, 8.21452),
            Point(6.15843, 7.38223),
            Point(9.15285, 7.50761),
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

    # Check attributes
    result_attrs = result_gdf.drop(columns="geometry").reset_index(drop=True)
    expected_attrs = expected_gdf.drop(columns="geometry").reset_index(drop=True)
    assert_frame_equal(result_attrs, expected_attrs)
