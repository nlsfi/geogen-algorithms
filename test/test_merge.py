#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from geogenalg import merge


@pytest.mark.parametrize(
    (
        "input_gdf",
        "attribute",
        "expected_num_geoms",
        "expected_attribute_values",
        "expected_geometries",
    ),
    [
        (
            gpd.GeoDataFrame(
                {
                    "class": ["a", "a", "b"],
                    "id": [1, 2, 3],
                },
                geometry=[
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(10, 10), (11, 11)]),
                ],
                crs="EPSG:3067",
            ),
            "class",
            2,
            {"a", "b"},
            [
                LineString([(0, 0), (1, 1), (2, 2)]),
                LineString([(10, 10), (11, 11)]),
            ],
        ),
        (
            gpd.GeoDataFrame(
                {
                    "class": ["b", "b"],
                    "id": [1, 2],
                },
                geometry=[
                    LineString([(0, 0), (1, 1)]),
                    LineString([(2, 1), (2, 2)]),
                ],
                crs="EPSG:3067",
            ),
            "class",
            2,
            {"b"},
            [
                LineString([(0, 0), (1, 1)]),
                LineString([(2, 1), (2, 2)]),
            ],
        ),
        (
            gpd.GeoDataFrame(
                {
                    "class": ["c", "c", "c"],
                    "id": [1, 2, 3],
                },
                geometry=[
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                    LineString([(2, 2), (3, 3)]),
                ],
                crs="EPSG:3067",
            ),
            "class",
            1,
            {"c"},
            [
                LineString([(0, 0), (1, 1), (2, 2), (3, 3)]),
            ],
        ),
        (
            gpd.GeoDataFrame(
                {
                    "class": ["x", "y"],
                    "id": [1, 2],
                },
                geometry=[
                    LineString([(0, 0), (1, 0)]),
                    LineString([(10, 10), (11, 11)]),
                ],
                crs="EPSG:3067",
            ),
            "class",
            2,
            {"x", "y"},
            [
                LineString([(0, 0), (1, 0)]),
                LineString([(10, 10), (11, 11)]),
            ],
        ),
    ],
    ids=[
        "merge_two_lines",
        "disjoint_lines_same_class",
        "merge_three_lines",
        "disjoint_lines_different_classes",
    ],
)
def test_merges_lines_with_same_attribute_value_into_one(
    input_gdf: gpd.GeoDataFrame,
    attribute: str,
    expected_num_geoms: int,
    expected_attribute_values: set[str],
    expected_geometries: list[LineString],
):
    result_gdf = merge.merge_connecting_lines_by_attribute(input_gdf, attribute)
    assert len(result_gdf) == expected_num_geoms
    assert set(result_gdf[attribute]) == expected_attribute_values
    assert all(geom.geom_type == "LineString" for geom in result_gdf.geometry)
    assert all(
        any(result_geom.equals(expected_geom) for result_geom in result_gdf.geometry)
        for expected_geom in expected_geometries
    )
