#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from numpy import nan
from shapely import LineString, MultiPolygon, Polygon, box, union_all

from geogenalg.transform import thin_polygon_sections_to_lines


@pytest.mark.parametrize(
    ("input_gdf", "expected", "threshold"),
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
                    "id": [3, 2, 1, 3],
                    "attribute": [
                        "section turns to line",
                        "turns to line",
                        "remains a polygon",
                        "section turns to line",
                    ],
                    "__old_ids": [
                        (3,),
                        (2,),
                        nan,
                        nan,
                    ],
                },
                geometry=[
                    LineString([[105, 133.49107142857142], [105, 237.5]]),
                    LineString([[55, 55], [55, 145]]),
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
        "no thin sections",
        "only thin sections",
        "mixed",
    ],
)
def test_thin_polygon_sections_to_lines(
    input_gdf: GeoDataFrame,
    expected: GeoDataFrame,
    threshold: float,
):
    input_gdf = input_gdf.set_index("id")
    modified = thin_polygon_sections_to_lines(
        input_gdf,
        threshold=threshold,
        min_line_length=10,
        min_new_section_length=10,
        min_new_section_area=250,
        old_ids_column="__old_ids",
    )
    expected = expected.set_index("id")

    assert_geodataframe_equal(modified, expected, check_like=True)
