#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from pathlib import Path

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString, Point, Polygon

from geogenalg.utility import dataframe_processing


def test_read_gdf_from_file_and_set_index(testdata_path: Path):
    file_name = testdata_path / "read_gdf.gpkg"

    gdf = dataframe_processing.read_gdf_from_file_and_set_index(
        file_name, "id", layer="read_gdf"
    )

    expected = GeoDataFrame(
        {"other_field": ["feature 1", "feature 2"]},
        index=[
            "8e7fae18-0a27-457b-995d-c545aa7feb35",
            "bf5223f9-e5e9-489f-b1e5-6009bc497471",
        ],
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:3067",
    )
    expected.index.name = "id"

    assert_geodataframe_equal(gdf, expected)


@pytest.mark.parametrize(
    ("input_gdfs", "expected_output"),
    [
        (
            [
                GeoDataFrame(
                    {"id": [1]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
                ),
                GeoDataFrame(
                    {"id": [2]}, geometry=[Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])]
                ),
            ],
            {
                "polygon_gdf": GeoDataFrame(
                    {"id": [1, 2]},
                    geometry=[
                        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                    ],
                ),
                "line_gdf": GeoDataFrame(columns=["geometry"], geometry="geometry"),
                "point_gdf": GeoDataFrame(columns=["geometry"], geometry="geometry"),
            },
        ),
        (
            [
                GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)]),
                GeoDataFrame({"id": [2]}, geometry=[LineString([(0, 0), (1, 1)])]),
                GeoDataFrame(
                    {"id": [3]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
                ),
            ],
            {
                "point_gdf": GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)]),
                "line_gdf": GeoDataFrame(
                    {"id": [2]}, geometry=[LineString([(0, 0), (1, 1)])]
                ),
                "polygon_gdf": GeoDataFrame(
                    {"id": [3]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
                ),
            },
        ),
        (
            [],
            {
                "polygon_gdf": GeoDataFrame(columns=["geometry"], geometry="geometry"),
                "line_gdf": GeoDataFrame(columns=["geometry"], geometry="geometry"),
                "point_gdf": GeoDataFrame(columns=["geometry"], geometry="geometry"),
            },
        ),
        (
            [
                GeoDataFrame(
                    {"id": [1]}, geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
                ),
                GeoDataFrame(
                    {"id": [2]}, geometry=[Polygon([(3, 3), (4, 3), (4, 4), (3, 4)])]
                ),
                GeoDataFrame(
                    {"id": [3]}, geometry=[Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])]
                ),
            ],
            {
                "polygon_gdf": GeoDataFrame(
                    {"id": [1, 2, 3]},
                    geometry=[
                        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                        Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
                        Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
                    ],
                ),
                "line_gdf": GeoDataFrame(columns=["geometry"], geometry="geometry"),
                "point_gdf": GeoDataFrame(columns=["geometry"], geometry="geometry"),
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
    input_gdfs: list[GeoDataFrame], expected_output: dict[str, GeoDataFrame]
):
    result_gdfs = dataframe_processing.group_gdfs_by_geometry_type(input_gdfs)

    for key in ["polygon_gdf", "line_gdf", "point_gdf"]:
        expected_gdf = expected_output[key]
        result_gdf = result_gdfs[key]

        expected_gdf = expected_gdf.reset_index(drop=True)
        result_gdf = result_gdf.reset_index(drop=True)

        assert_frame_equal(result_gdf, expected_gdf, check_dtype=False)
