#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from geogenalg.cluster import group_touching_or_intersecting_features


@pytest.mark.parametrize(
    ("geometries", "expected_groups"),
    [
        (
            [LineString([(0, 0), (1, 1)])],
            [0],
        ),
        (
            [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
            [0],
        ),
        (
            [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 1)])],
            [0],
        ),
        (
            [LineString([(0, 0), (1, 1)]), LineString([(2, 1), (3, 1)])],
            [0, 1],
        ),
        (
            [],
            [],
        ),
        (
            [Point(0, 0)],
            [0],
        ),
        (
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
                Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
            ],
            [0, 1],
        ),
    ],
    ids=[
        "single line",
        "touching lines",
        "overlapping lines",
        "disjoint lines",
        "empty data",
        "single point",
        "two groups of overlapping polygons",
    ],
)
def test_group_touching_or_intersecting(
    geometries: list[BaseGeometry], expected_groups: list[int]
):
    data = GeoDataFrame.from_dict(
        {"id": list(range(len(geometries))), "geometry": geometries},
        crs="EPSG:3067",
    )
    group_touching_or_intersecting_features(data)

    if len(expected_groups) == 0:
        assert data.empty
    else:
        assert sorted(data["_group"].unique()) == expected_groups
