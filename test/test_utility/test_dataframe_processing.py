#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString, Point, Polygon

from geogenalg.utility import dataframe_processing


@pytest.mark.parametrize(
    ("input_gdfs", "expected_output"),
    [
        (
            [
                gpd.GeoDataFrame(
                    {"id": [1]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
                ),
                gpd.GeoDataFrame(
                    {"id": [2]}, geometry=[Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])]
                ),
            ],
            {
                "polygon_gdf": gpd.GeoDataFrame(
                    {"id": [1, 2]},
                    geometry=[
                        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                    ],
                ),
                "line_gdf": gpd.GeoDataFrame(columns=["geometry"], geometry="geometry"),
                "point_gdf": gpd.GeoDataFrame(
                    columns=["geometry"], geometry="geometry"
                ),
            },
        ),
        (
            [
                gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)]),
                gpd.GeoDataFrame({"id": [2]}, geometry=[LineString([(0, 0), (1, 1)])]),
                gpd.GeoDataFrame(
                    {"id": [3]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
                ),
            ],
            {
                "point_gdf": gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)]),
                "line_gdf": gpd.GeoDataFrame(
                    {"id": [2]}, geometry=[LineString([(0, 0), (1, 1)])]
                ),
                "polygon_gdf": gpd.GeoDataFrame(
                    {"id": [3]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
                ),
            },
        ),
        (
            [],
            {
                "polygon_gdf": gpd.GeoDataFrame(
                    columns=["geometry"], geometry="geometry"
                ),
                "line_gdf": gpd.GeoDataFrame(columns=["geometry"], geometry="geometry"),
                "point_gdf": gpd.GeoDataFrame(
                    columns=["geometry"], geometry="geometry"
                ),
            },
        ),
        (
            [
                gpd.GeoDataFrame(
                    {"id": [1]}, geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
                ),
                gpd.GeoDataFrame(
                    {"id": [2]}, geometry=[Polygon([(3, 3), (4, 3), (4, 4), (3, 4)])]
                ),
                gpd.GeoDataFrame(
                    {"id": [3]}, geometry=[Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])]
                ),
            ],
            {
                "polygon_gdf": gpd.GeoDataFrame(
                    {"id": [1, 2, 3]},
                    geometry=[
                        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                        Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
                        Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
                    ],
                ),
                "line_gdf": gpd.GeoDataFrame(columns=["geometry"], geometry="geometry"),
                "point_gdf": gpd.GeoDataFrame(
                    columns=["geometry"], geometry="geometry"
                ),
            },
        ),
    ],
    ids=[
        "two polygon GeoDataFrames should merge",
        "one input GeoDataFrame for each geometry type",
        "empty input list should return empty GeoDataFrames",
        "three polygon GeoDataFrames should merge",
    ],
)
def test_group_gdfs_by_geometry_type(
    input_gdfs: list[gpd.GeoDataFrame], expected_output: dict[str, gpd.GeoDataFrame]
):
    result_gdfs = dataframe_processing.group_gdfs_by_geometry_type(input_gdfs)

    for key in ["polygon_gdf", "line_gdf", "point_gdf"]:
        expected_gdf = expected_output[key]
        result_gdf = result_gdfs[key]

        expected_gdf = expected_gdf.reset_index(drop=True)
        result_gdf = result_gdf.reset_index(drop=True)

        assert_frame_equal(result_gdf, expected_gdf, check_dtype=False)
