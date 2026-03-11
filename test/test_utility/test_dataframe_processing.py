#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
import re
from pathlib import Path

import pytest
from geopandas import GeoDataFrame
from geopandas.geoseries import GeoSeries
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString, Point, Polygon

from geogenalg.core.exceptions import GeoCombineError
from geogenalg.utility import dataframe_processing
from geogenalg.utility.dataframe_processing import (
    ConcatParameters,
    combine_gdfs,
    combine_geoseries,
)


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


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (
            [DataFrame()],
            "Non-GeoDataFrame object found.",
        ),
        (
            [GeoDataFrame(geometry=[], crs="EPSG:3857"), DataFrame()],
            "Non-GeoDataFrame object found.",
        ),
        (
            [GeoDataFrame()],
            "GeoDataFrame does not have active geometry column set.",
        ),
        (
            [
                GeoDataFrame(geometry=[], crs="EPSG:3857"),
                GeoDataFrame(geometry=[], crs="EPSG:4326"),
            ],
            "Different CRSs found in GeoDataFrames.",
        ),
        (
            [
                GeoDataFrame(
                    {
                        "geom": [],
                    },
                    geometry="geom",
                    crs="EPSG:3857",
                ),
                GeoDataFrame(
                    {
                        "geometry": [],
                    },
                    geometry="geometry",
                    crs="EPSG:3857",
                ),
            ],
            "Different geometry column names found in GeoDataFrames.",
        ),
    ],
    ids=[
        "dataframe",
        "gdf_mixed_with_df",
        "no_geom_column",
        "different_crss",
        "different_geom_col_names",
    ],
)
def test_combine_gdfs_raises(
    inputs: list[GeoDataFrame],
    message: str,
):
    with pytest.raises(GeoCombineError, match=re.escape(message)):
        combine_gdfs(inputs)


@pytest.mark.parametrize(
    ("inputs", "kwargs", "output"),
    [
        (
            [
                GeoDataFrame(geometry=[], crs="EPSG:3857"),
            ],
            {},
            GeoDataFrame(geometry=[], crs="EPSG:3857"),
        ),
        (
            [
                GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3857"),
            ],
            {},
            GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3857"),
        ),
        (
            [
                GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:3857"),
                GeoDataFrame(geometry=[Point(1, 0)], crs="EPSG:3857"),
            ],
            {},
            GeoDataFrame(
                index=[0, 0],
                geometry=[Point(0, 0), Point(1, 0)],
                crs="EPSG:3857",
            ),
        ),
        (
            [
                GeoDataFrame(
                    index=[4],
                    geometry=[Point(0, 0)],
                    crs="EPSG:3857",
                ),
                GeoDataFrame(
                    index=[7],
                    geometry=[Point(1, 0)],
                    crs="EPSG:3857",
                ),
            ],
            {"ignore_index": True},
            GeoDataFrame(
                index=[0, 1],
                geometry=[Point(0, 0), Point(1, 0)],
                crs="EPSG:3857",
            ),
        ),
    ],
    ids=[
        "unchanged_no_features",
        "unchanged_has_features",
        "gets_combined",
        "kwargs",
    ],
)
def test_combine_gdfs_returns(
    inputs: list[GeoDataFrame],
    kwargs: ConcatParameters,
    output: GeoDataFrame,
):
    assert_geodataframe_equal(
        combine_gdfs(inputs, **kwargs),
        output,
    )


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (
            [Series()],
            "Non-GeoSeries object found.",
        ),
        (
            [GeoSeries(crs="EPSG:3857"), Series()],
            "Non-GeoSeries object found.",
        ),
        (
            [
                GeoSeries(crs="EPSG:3857"),
                GeoSeries(crs="EPSG:4326"),
            ],
            "Different CRSs found in GeoDataFrames.",
        ),
        (
            [
                GeoSeries(
                    name="geom",
                    crs="EPSG:3857",
                ),
                GeoSeries(
                    name="geometry",
                    crs="EPSG:3857",
                ),
            ],
            "Different geometry column names found in GeoDataFrames.",
        ),
    ],
    ids=[
        "series",
        "series_mixed_with_geoseries",
        "different_crss",
        "different_geom_col_names",
    ],
)
def test_combine_geoseries_raises(
    inputs: list[GeoSeries],
    message: str,
):
    with pytest.raises(GeoCombineError, match=re.escape(message)):
        combine_geoseries(inputs)


@pytest.mark.parametrize(
    ("inputs", "kwargs", "output"),
    [
        (
            [GeoSeries(crs="EPSG:3857")],
            {},
            GeoSeries(crs="EPSG:3857"),
        ),
        (
            [
                GeoSeries([Point(0, 0)], crs="EPSG:3857"),
            ],
            {},
            GeoSeries([Point(0, 0)], crs="EPSG:3857"),
        ),
        (
            [
                GeoSeries([Point(0, 0)], crs="EPSG:3857"),
                GeoSeries([Point(1, 0)], crs="EPSG:3857"),
            ],
            {},
            GeoSeries(
                [Point(0, 0), Point(1, 0)],
                index=[0, 0],
                crs="EPSG:3857",
            ),
        ),
        (
            [
                GeoSeries(
                    [Point(0, 0)],
                    index=[4],
                    crs="EPSG:3857",
                ),
                GeoSeries(
                    [Point(1, 0)],
                    index=[7],
                    crs="EPSG:3857",
                ),
            ],
            {"ignore_index": True},
            GeoSeries(
                [Point(0, 0), Point(1, 0)],
                index=[0, 1],
                crs="EPSG:3857",
            ),
        ),
    ],
    ids=[
        "unchanged_no_features",
        "unchanged_has_features",
        "gets_combined",
        "kwargs",
    ],
)
def test_combine_geries_returns(
    inputs: list[GeoSeries],
    kwargs: ConcatParameters,
    output: GeoSeries,
):
    assert_geoseries_equal(
        combine_geoseries(inputs, **kwargs),
        output,
    )
