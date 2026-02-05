#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from numpy import nan
from shapely import LineString, MultiPolygon, Polygon, box, union_all

from geogenalg.transform import thin_polygon_sections_to_lines


@pytest.mark.parametrize(
    ("input_gdf", "expected_lines", "expected_polygons", "threshold"),
    [
        (
            GeoDataFrame(
                {
                    "id": [1],
                    "attribute": ["attribute"],
                },
                geometry=[
                    box(0, 0, 100, 20),
                ],
            ),
            GeoDataFrame(columns=["id", "attribute", "__old_ids"], geometry=[]),
            GeoDataFrame(
                {
                    "id": [1],
                    "attribute": ["attribute"],
                    "__olds_ids": [nan],
                },
                geometry=[
                    box(0, 0, 100, 20),
                ],
            ),
            10,
        ),
        (
            GeoDataFrame(
                {
                    "id": [1],
                    "attribute": ["attribute"],
                },
                geometry=[
                    box(0, 0, 100, 20),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1],
                    "attribute": ["attribute"],
                    "__old_ids": [(1,)],
                },
                geometry=[LineString([[10, 10], [90, 10]])],
            ),
            GeoDataFrame(columns=["id", "attribute", "__old_ids"], geometry=[]),
            100,
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2, 3],
                    "attribute": [
                        "remains a polygon",
                        "turns to line",
                        "section turns to line",
                    ],
                },
                geometry=[
                    box(0, 0, 100, 20),
                    box(50, 50, 60, 150),
                    union_all(
                        [
                            box(80, 30, 150, 130),
                            box(102.5, 130, 107.5, 240),
                        ]
                    ),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [3, 2],
                    "attribute": [
                        "section turns to line",
                        "turns to line",
                    ],
                    "__old_ids": [
                        (3,),
                        (2,),
                    ],
                },
                geometry=[
                    LineString([[105, 133.49107142857142], [105, 237.5]]),
                    LineString([[55, 55], [55, 145]]),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1, 3],
                    "attribute": [
                        "remains a polygon",
                        "section turns to line",
                    ],
                    "__old_ids": [
                        None,
                        None,
                    ],
                },
                geometry=[
                    box(0, 0, 100, 20),
                    MultiPolygon(
                        [
                            Polygon(
                                [
                                    [150, 30],
                                    [80, 30],
                                    [80, 130],
                                    [102.5, 130],
                                    [102.5, 133.49107142857142],
                                    [107.5, 133.49107142857142],
                                    [107.5, 130],
                                    [150, 130],
                                    [150, 30],
                                ]
                            )
                        ]
                    ),
                ],
            ),
            15,
        ),
    ],
    ids=[
        "no_thin_sections",
        "only_thin_sections",
        "mixed",
    ],
)
def test_thin_polygon_sections_to_lines(
    input_gdf: GeoDataFrame,
    expected_lines: GeoDataFrame,
    expected_polygons: GeoDataFrame,
    threshold: float,
):
    input_gdf = input_gdf.set_index("id")
    modified_lines, modified_polygons = thin_polygon_sections_to_lines(
        input_gdf,
        threshold=threshold,
        min_line_length=10,
        min_new_section_length=10,
        min_new_section_area=250,
        old_ids_column="__old_ids",
    )
    expected_lines = expected_lines.set_index("id")
    expected_polygons = expected_polygons.set_index("id")

    assert_geodataframe_equal(modified_lines, expected_lines, check_like=True)
    assert_geodataframe_equal(modified_polygons, expected_polygons, check_like=True)
