#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import re

import numpy as np
import pytest
from geopandas import GeoDataFrame
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString, Point, Polygon

from geogenalg.analyze import (
    calculate_coverage,
    calculate_main_angle,
    classify_polygons_by_size_of_minimum_bounding_rectangle,
    flag_parallel_lines,
    get_parallel_line_areas,
)
from geogenalg.core.exceptions import GeometryTypeError


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


@pytest.mark.parametrize(
    ("polygon", "expected_angles"),
    [
        (Polygon([(0, 0), (1, 0), (1, 2), (0, 2)]), [0.0]),
        (
            Polygon([(0, 0), (1, 1), (0, 2), (-1, 1)]),
            [45.0, 135.0],
        ),
        (
            Polygon([(0, 0), (4, 0), (4, 1), (0, 1)]),
            [90.0],
        ),
        (
            Polygon(
                [
                    (0.0, 0.0),
                    (3.759525, 1.368080),
                    (3.098076, 3.129410),
                    (-0.661449, 1.761330),
                ]
            ),
            [70.0],
        ),
        (
            Polygon(
                [
                    (4.0, 2.0),
                    (3.0, 4.0),
                    (1.0, 3.0),
                    (0.0, 5.0),
                    (-2.0, 4.0),
                    (-1.0, 2.0),
                    (-3.0, 1.0),
                    (-2.0, -1.0),
                ]
            ),
            [63.434],
        ),
    ],
    ids=[
        "axis-aligned rectangle",
        "rotated rectangle (45 degrees)",
        "horizontal elongated",
        "70° rotated rectangle",
        "rectangle with bump should return the main orientation angle",
    ],
)
def test_calculate_main_angle_valid(polygon: Polygon, expected_angles: list[float]):
    angle = calculate_main_angle(polygon)
    assert any(
        angle == pytest.approx(expected, abs=0.01) for expected in expected_angles
    )


def test_calculate_main_angle_invalid_geometry():
    with pytest.raises(TypeError, match="Input geometry must be a Shapely Polygon"):
        calculate_main_angle(LineString([(0, 0), (1, 1)]))


def test_calculate_main_angle_empty_polygon():
    empty_poly = Polygon()
    result = calculate_main_angle(empty_poly)
    assert np.isnan(result)


@pytest.mark.parametrize(
    (
        "input_gdf",
        "side_threshold",
        "expected_gdfs",
    ),
    [
        (
            GeoDataFrame(
                {"id": [1], "class": ["A"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            ),
            5.0,
            {
                "small_polygons": GeoDataFrame(
                    {"id": [1], "class": ["A"]},
                    geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                ),
                "large_polygons": GeoDataFrame(columns=["id", "class"], geometry=[]),
            },
        ),
        (
            GeoDataFrame(
                {"id": [1], "class": ["A"]},
                geometry=[Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])],
            ),
            5.0,
            {
                "small_polygons": GeoDataFrame(columns=["id", "class"], geometry=[]),
                "large_polygons": GeoDataFrame(
                    {"id": [1], "class": ["A"]},
                    geometry=[Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])],
                ),
            },
        ),
        (
            GeoDataFrame(
                {"id": [1], "class": ["SPECIAL"]},
                geometry=[Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])],
            ),
            15.0,
            {
                "small_polygons": GeoDataFrame(
                    {"id": [1], "class": ["SPECIAL"]},
                    geometry=[Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])],
                ),
                "large_polygons": GeoDataFrame(columns=["id", "class"], geometry=[]),
            },
        ),
        (
            GeoDataFrame(columns=["id", "class"], geometry=[]),
            5.0,
            {
                "small_polygons": GeoDataFrame(columns=["id", "class"], geometry=[]),
                "large_polygons": GeoDataFrame(columns=["id", "class"], geometry=[]),
            },
        ),
        (
            GeoDataFrame(
                {"id": [1, 2], "class": ["A", "B"]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(0, 0), (10, 0), (10, 1), (0, 1)]),
                ],
            ),
            5.0,
            {
                "small_polygons": GeoDataFrame(
                    {"id": [1], "class": ["A"]},
                    geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                ),
                "large_polygons": GeoDataFrame(
                    {"id": [2], "class": ["B"]},
                    geometry=[Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])],
                ),
            },
        ),
        (
            GeoDataFrame(
                {"id": [1], "class": ["A"]},
                geometry=[
                    Polygon(
                        [
                            (0, 0),
                            (1, 1),
                            (2, 0),
                            (3, 1),
                            (6, 0),
                            (3, -1),
                            (2, -0.5),
                            (1, -1),
                            (0, 0),
                        ]
                    )
                ],
            ),
            5.5,
            {
                "small_polygons": GeoDataFrame(columns=["id", "class"], geometry=[]),
                "large_polygons": GeoDataFrame(
                    {"id": [1], "class": ["A"]},
                    geometry=[
                        Polygon(
                            [
                                (0, 0),
                                (1, 1),
                                (2, 0),
                                (3, 1),
                                (6, 0),
                                (3, -1),
                                (2, -0.5),
                                (1, -1),
                                (0, 0),
                            ]
                        )
                    ],
                ),
            },
        ),
    ],
    ids=[
        "single small polygon",
        "single large polygon",
        "large polygon with attributes",
        "empty GeoDataFrame",
        "mix of large and small polygons",
        "irregular polygon with small edge segments",
    ],
)
def test_classify_polygons_by_size_of_minimum_bounding_rectangle(
    input_gdf: GeoDataFrame,
    side_threshold: float,
    expected_gdfs: dict[str, GeoDataFrame],
):
    result_gdfs = classify_polygons_by_size_of_minimum_bounding_rectangle(
        input_gdf,
        side_threshold,
    )

    for key, expected in expected_gdfs.items():
        result_sorted = (
            result_gdfs[key]
            .sort_values(list(expected.columns.drop("geometry")))
            .reset_index(drop=True)
        )
        expected_sorted = expected.sort_values(
            list(expected.columns.drop("geometry"))
        ).reset_index(drop=True)

        for result_geom, expected_geom in zip(
            result_sorted.geometry, expected_sorted.geometry, strict=False
        ):
            assert result_geom.equals(expected_geom), (
                f"{key} geometries differ: {result_geom} vs {expected_geom}"
            )

        result_attrs = result_sorted.drop(columns="geometry")
        expected_attrs = expected_sorted.drop(columns="geometry")
        assert_frame_equal(result_attrs, expected_attrs, check_dtype=False)


def test_classify_polygons_by_size_of_minimum_bounding_rectangle_raises_on_invalid_geometry():
    invalid_gdf = GeoDataFrame(
        {"id": [1, 2, 3], "class": ["A", "B", "C"]},
        geometry=[
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ],
    )

    with pytest.raises(
        GeometryTypeError,
        match=re.escape(
            "Classify polygons only supports Polygon or MultiPolygon geometries."
        ),
    ):
        classify_polygons_by_size_of_minimum_bounding_rectangle(invalid_gdf, 5.0)


@pytest.mark.parametrize(
    (
        "input_gdf",
        "parallel_distance",
        "allowed_direction_difference",
        "segmentize_distance",
        "expected",
    ),
    [
        (  # empty
            GeoDataFrame(geometry=[]),
            1,
            10,
            0,
            GeoDataFrame(
                {
                    "parallel_group": [],
                    "__direction": [],
                    "parallel_with": [],
                    "__id": [],
                },
                geometry=[],
            ),
        ),
        (  # no_parallel_lines
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                ]
            ),
            1,
            10,
            0,
            GeoDataFrame(
                {
                    "parallel_group": [],
                    "__direction": [],
                    "parallel_with": [],
                    "__id": [],
                },
                geometry=[],
            ),
        ),
        (  # two_parallel_lines
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 1], [1, 1]]),
                ]
            ),
            1,
            10,
            0,
            GeoDataFrame(
                {
                    "parallel_group": [1, 1],
                    "__direction": [270.0, 270.0],
                    "parallel_with": [{1}, {0}],
                    "__id": [0, 1],
                },
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 1], [1, 1]]),
                ],
            ),
        ),
        (  # direction_difference_too_large
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 1], [1, 0.5]]),
                ]
            ),
            1,
            10,
            0,
            GeoDataFrame(
                {
                    "parallel_group": [],
                    "__direction": [],
                    "parallel_with": [],
                    "__id": [],
                },
                geometry=[],
            ),
        ),
        (  # in_direction_difference_bounds
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 2], [1, 1]]),
                ]
            ),
            5,
            50,
            0,
            GeoDataFrame(
                {
                    "parallel_group": [1, 1],
                    "__direction": [270.0, 315.0],
                    "parallel_with": [{1}, {0}],
                    "__id": [0, 1],
                },
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 2], [1, 1]]),
                ],
            ),
        ),
        (  # multiple_parallel_lines
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 1], [1, 1]]),
                    LineString([[0, 2], [1, 2]]),
                    LineString([[0, 20], [1, 20]]),
                    LineString([[0, 21], [1, 21]]),
                ]
            ),
            5,
            10,
            0,
            GeoDataFrame(
                {
                    "parallel_group": [1, 1, 1, 2, 2],
                    "__direction": [270.0, 270.0, 270.0, 270.0, 270.0],
                    "parallel_with": [{1, 2}, {0, 2}, {1, 0}, {4}, {3}],
                    "__id": [0, 1, 2, 3, 4],
                },
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 1], [1, 1]]),
                    LineString([[0, 2], [1, 2]]),
                    LineString([[0, 20], [1, 20]]),
                    LineString([[0, 21], [1, 21]]),
                ],
            ),
        ),
    ],
    ids=[
        "empty",
        "no_parallel_lines",
        "two_parallel_lines",
        "direction_difference_too_large",
        "in_direction_difference_bounds",
        "multiple_parallel_lines",
    ],
)
def test_flag_parallel_lines(
    input_gdf: GeoDataFrame,
    parallel_distance: float,
    allowed_direction_difference: float,
    segmentize_distance: float,
    expected: GeoDataFrame,
):
    if expected.empty:
        expected["__direction"] = expected["__direction"].astype("float64")
        expected["parallel_with"] = expected["parallel_with"].astype("object")
        expected["parallel_group"] = expected["parallel_group"].astype("int64")
        expected["__id"] = expected["__id"].astype("int64")

    result = flag_parallel_lines(
        input_gdf,
        parallel_distance,
        allowed_direction_difference,
        segmentize_distance=segmentize_distance,
    )

    assert_frame_equal(
        result,
        expected,
        check_like=True,
    )


@pytest.mark.parametrize(
    (
        "input_gdf",
        "parallel_distance",
        "allowed_direction_difference",
        "segmentize_distance",
        "expected",
    ),
    [
        (  # empty
            GeoDataFrame(geometry=[]),
            1,
            10,
            0,
            GeoDataFrame(),
        ),
        (  # no_parallel_lines
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                ]
            ),
            1,
            10,
            0,
            GeoDataFrame(),
        ),
        (  # two_parallel_lines
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 1], [1, 1]]),
                ]
            ),
            1,
            10,
            0,
            GeoDataFrame(
                {
                    "__direction": [270.0],
                },
                geometry=[
                    Polygon(
                        [
                            [0, 0],
                            [0, 1],
                            [1, 1],
                            [1, 0],
                            [0, 0],
                        ]
                    ),
                ],
            ),
        ),
        (  # direction_difference_too_large
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 1], [1, 0.5]]),
                ]
            ),
            1,
            10,
            0,
            GeoDataFrame(),
        ),
        (  # in_direction_difference_bounds
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 2], [1, 1]]),
                ]
            ),
            5,
            50,
            0,
            GeoDataFrame(
                {
                    "__direction": [290.0],
                },
                geometry=[
                    Polygon(
                        [
                            [0, 2],
                            [1, 1],
                            [1, 0],
                            [0, 0],
                            [0, 2],
                        ]
                    ),
                ],
            ),
        ),
        (  # multiple_parallel_lines
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [1, 0]]),
                    LineString([[0, 1], [1, 1]]),
                    LineString([[0, 2], [1, 2]]),
                    LineString([[0, 20], [1, 20]]),
                    LineString([[0, 21], [1, 21]]),
                ]
            ),
            5,
            10,
            0,
            GeoDataFrame(
                {
                    "__direction": [270.0, 270.0],
                },
                geometry=[
                    Polygon(
                        [
                            [1, 2],
                            [1, 1],
                            [1, 0],
                            [0, 0],
                            [0, 1],
                            [0, 2],
                            [1, 2],
                        ]
                    ),
                    Polygon(
                        [
                            [0, 21],
                            [1, 21],
                            [1, 20],
                            [0, 20],
                            [0, 21],
                        ]
                    ),
                ],
            ),
        ),
    ],
    ids=[
        "empty",
        "no_parallel_lines",
        "two_parallel_lines",
        "direction_difference_too_large",
        "in_direction_difference_bounds",
        "multiple_parallel_lines",
    ],
)
def test_get_parallel_line_areas(
    input_gdf: GeoDataFrame,
    parallel_distance: float,
    allowed_direction_difference: float,
    segmentize_distance: float,
    expected: GeoDataFrame,
):
    result = get_parallel_line_areas(
        input_gdf,
        parallel_distance,
        allowed_direction_difference=allowed_direction_difference,
        segmentize_distance=segmentize_distance,
    )

    assert_frame_equal(
        result,
        expected,
        check_like=True,
    )
