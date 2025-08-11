#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

from geogenalg.analyze import calculate_coverage


def test_calculate_coverage():
    test_features = calculate_coverage(
        overlay_features=GeoDataFrame(
            {
                "id": [1],
            },
            geometry=[
                Polygon([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]),
            ],
            crs="EPSG:3067",
        ),
        base_features=GeoDataFrame(
            {
                "class": ["a"],
                "id": [1],
            },
            geometry=[
                Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            ],
            crs="EPSG:3067",
        ),
    )
    assert "coverage" in test_features.columns
    assert test_features["coverage"][0] == pytest.approx(56.25)


def test_calculate_coverage_raises_for_wrong_crs_type():
    with pytest.raises(ValueError, match="projected CRS"):
        calculate_coverage(
            overlay_features=GeoDataFrame(
                {
                    "id": [1],
                },
                geometry=[
                    Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
                ],
            ),
            base_features=GeoDataFrame(
                {
                    "class": ["a"],
                    "id": [1],
                },
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                ],
            ),
        )


def test_calculate_coverage_using_empty_input():
    overlay = GeoDataFrame({"id": []}, geometry=[], crs="EPSG:3067")
    base = GeoDataFrame({"class": [], "id": []}, geometry=[], crs="EPSG:3067")

    result = calculate_coverage(overlay, base)

    assert result.empty
    assert "coverage" in result.columns
