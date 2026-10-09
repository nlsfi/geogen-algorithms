#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


import pytest
from geopandas.geodataframe import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from pandas import Series
from shapely.geometry import LineString

from geogenalg.application.generalize_watercourse_lines import (
    GeneralizeWaterCourseLines,
)


@pytest.mark.parametrize(
    ("input_gdf", "expected_flags"),
    [
        pytest.param(
            GeoDataFrame(
                {"parallel_group": Series([], dtype=int)},
                geometry=[],
            ),
            Series([], dtype=bool),
            id="empty",
        ),
        pytest.param(
            GeoDataFrame(
                {"parallel_group": [-1, -1]},
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (100, 10)]),
                ],
            ),
            Series([False, False], dtype=bool),
            id="all_unassigned_ignored",
        ),
        pytest.param(
            GeoDataFrame(
                {"parallel_group": [0]},
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                ],
            ),
            Series([True], dtype=bool),
            id="single_horizontal_line_hits_bisector",
        ),
        pytest.param(
            GeoDataFrame(
                {"parallel_group": [0, 0]},
                geometry=[
                    LineString([(0, 0), (50, 0)]),
                    LineString([(50, 0), (100, 0)]),
                ],
            ),
            Series([True, True], dtype=bool),
            id="touching_collinear_chain_all_flagged",
        ),
        pytest.param(
            GeoDataFrame(
                {"parallel_group": [0, 0]},
                geometry=[
                    LineString([(0, 0), (50, 0)]),
                    LineString([(0, 10), (50, 10)]),
                ],
            ),
            Series([True, True], dtype=bool),
            id="parallel_pair_both_flagged",
        ),
        pytest.param(
            GeoDataFrame(
                {"parallel_group": [0, 0, 0]},
                geometry=[
                    LineString([(0, 0), (50, 0)]),
                    LineString([(50, 0), (100, 0)]),
                    LineString([(0, 10), (100, 10)]),
                ],
            ),
            Series([True, True, True], dtype=bool),
            id="group_with_crossing_strand_flags_entire_group",
        ),
        pytest.param(
            GeoDataFrame(
                {"parallel_group": [0, 0, 1, 1]},
                geometry=[
                    LineString([(0, 0), (50, 0)]),
                    LineString([(50, 0), (100, 0)]),
                    LineString([(0, 10), (50, 10)]),
                    LineString([(50, 10), (100, 10)]),
                ],
            ),
            Series([True, True, True, True], dtype=bool),
            id="multiple_groups_processed_independently",
        ),
    ],
)
def test_evaluate_perpendicular_intersections(
    input_gdf: GeoDataFrame,
    expected_flags: Series,
):
    result = GeneralizeWaterCourseLines._evaluate_perpendicular_intersections(input_gdf)

    assert "intersects_bisector" in result.columns
    assert list(result["intersects_bisector"]) == list(expected_flags)


@pytest.mark.parametrize(
    ("input_gdf", "expected"),
    [
        pytest.param(
            GeoDataFrame(geometry=[]),
            GeoDataFrame(
                {"parallel_group": Series([], dtype=int)},
                geometry=[],
            ),
            id="empty",
        ),
        pytest.param(
            GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])]),
            GeoDataFrame(
                {"parallel_group": [-1]},
                geometry=[LineString([(0, 0), (100, 0)])],
            ),
            id="single_line",
        ),
        pytest.param(
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (100, 10)]),
                ],
            ),
            GeoDataFrame(
                {"parallel_group": [0, 0]},
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (100, 10)]),
                ],
            ),
            id="parallel_pair_grouped",
        ),
        pytest.param(
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 100), (100, 100)]),
                ],
            ),
            GeoDataFrame(
                {"parallel_group": [-1, -1]},
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 100), (100, 100)]),
                ],
            ),
            id="too_far_apart",
        ),
        pytest.param(
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (20, 10)]),
                ],
            ),
            GeoDataFrame(
                {"parallel_group": [-1, -1]},
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (20, 10)]),
                ],
            ),
            id="short_line_misses_bisector_group_dropped",
        ),
        pytest.param(
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (100, 10)]),
                    LineString([(0, 20), (20, 20)]),
                ],
            ),
            GeoDataFrame(
                {"parallel_group": [0, 0, -1]},
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (100, 10)]),
                    LineString([(0, 20), (20, 20)]),
                ],
            ),
            id="line_missing_bisector_evicted",
        ),
        pytest.param(
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (100, 10)]),
                    LineString([(0, 500), (100, 500)]),
                    LineString([(0, 510), (100, 510)]),
                ],
            ),
            GeoDataFrame(
                {"parallel_group": [0, 0, 1, 1]},
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (100, 10)]),
                    LineString([(0, 500), (100, 500)]),
                    LineString([(0, 510), (100, 510)]),
                ],
            ),
            id="separate_groups_numbered",
        ),
        pytest.param(
            GeoDataFrame(
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (100, 10)]),
                ],
                index=[5, 7],
            ),
            GeoDataFrame(
                {"parallel_group": [0, 0]},
                geometry=[
                    LineString([(0, 0), (100, 0)]),
                    LineString([(0, 10), (100, 10)]),
                ],
            ),
            id="index_reset",
        ),
    ],
)
def test_refine_parallel_groups(
    input_gdf: GeoDataFrame,
    expected: GeoDataFrame,
):
    result = GeneralizeWaterCourseLines._refine_parallel_groups(
        input_gdf,
        parallel_distance=45.0,
        allowed_direction_difference=10.0,
    )
    assert_geodataframe_equal(result, expected, check_like=True)
