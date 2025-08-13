#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import re

import geopandas as gpd
import pytest
from geopandas import GeoDataFrame
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString, Point, Polygon

from geogenalg.analyze import (
    calculate_coverage,
    calculate_main_angle,
    classify_polygons_by_size_of_minimum_bounding_rectangle,
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
    with pytest.raises(ValueError, match="Input polygon is empty"):
        calculate_main_angle(empty_poly)


@pytest.mark.parametrize(
    (
        "input_gdf",
        "side_threshold",
        "class_column",
        "classes_for_ignore_size",
        "expected_gdfs",
    ),
    [
        (
            gpd.GeoDataFrame(
                {"id": [1], "class": ["A"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            ),
            5.0,
            "class",
            [],
            {
                "small_polygons": gpd.GeoDataFrame(
                    {"id": [1], "class": ["A"]},
                    geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                ),
                "large_polygons": gpd.GeoDataFrame(
                    columns=["id", "class"], geometry=[]
                ),
            },
        ),
        (
            gpd.GeoDataFrame(
                {"id": [1], "class": ["A"]},
                geometry=[Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])],
            ),
            5.0,
            "class",
            [],
            {
                "small_polygons": gpd.GeoDataFrame(
                    columns=["id", "class"], geometry=[]
                ),
                "large_polygons": gpd.GeoDataFrame(
                    {"id": [1], "class": ["A"]},
                    geometry=[Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])],
                ),
            },
        ),
        (
            gpd.GeoDataFrame(
                {"id": [1], "class": ["SPECIAL"]},
                geometry=[Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])],
            ),
            5.0,
            "class",
            ["SPECIAL"],
            {
                "small_polygons": gpd.GeoDataFrame(
                    {"id": [1], "class": ["SPECIAL"]},
                    geometry=[Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])],
                ),
                "large_polygons": gpd.GeoDataFrame(
                    columns=["id", "class"], geometry=[]
                ),
            },
        ),
        (
            gpd.GeoDataFrame(columns=["id", "class"], geometry=[]),
            5.0,
            "class",
            [],
            {
                "small_polygons": gpd.GeoDataFrame(
                    columns=["id", "class"], geometry=[]
                ),
                "large_polygons": gpd.GeoDataFrame(
                    columns=["id", "class"], geometry=[]
                ),
            },
        ),
        (
            gpd.GeoDataFrame(
                {"id": [1, 2], "class": ["A", "B"]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(0, 0), (10, 0), (10, 1), (0, 1)]),
                ],
            ),
            5.0,
            "class",
            [],
            {
                "small_polygons": gpd.GeoDataFrame(
                    {"id": [1], "class": ["A"]},
                    geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                ),
                "large_polygons": gpd.GeoDataFrame(
                    {"id": [2], "class": ["B"]},
                    geometry=[Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])],
                ),
            },
        ),
        (
            gpd.GeoDataFrame(
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
            "class",
            [],
            {
                "small_polygons": gpd.GeoDataFrame(
                    columns=["id", "class"], geometry=[]
                ),
                "large_polygons": gpd.GeoDataFrame(
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
        "large polygon in ignore list",
        "empty GeoDataFrame",
        "mix of large and small polygons",
        "irregular polygon with small edge segments",
    ],
)
def test_classify_polygons_by_size_of_minimum_bounding_rectangle(
    input_gdf: gpd.GeoDataFrame,
    side_threshold: float,
    class_column: str,
    classes_for_ignore_size: list[str] | list[int],
    expected_gdfs: dict[str, gpd.GeoDataFrame],
):
    result_gdfs = classify_polygons_by_size_of_minimum_bounding_rectangle(
        input_gdf,
        side_threshold,
        class_column,
        classes_for_ignore_size,
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
    invalid_gdf = gpd.GeoDataFrame(
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
        classify_polygons_by_size_of_minimum_bounding_rectangle(
            invalid_gdf, 5.0, "class", []
        )
