#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from conftest import IntegrationTest
from geopandas import GeoDataFrame
from numpy import nan
from pandas import isna
from pandas.testing import assert_frame_equal
from shapely import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from geogenalg.application.generalize_buildings import GeneralizeBuildings
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.testing import (
    GeoPackagePath,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
from cartagen.algorithms import buildings

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_buildings_50k(testdata_path: Path) -> None:
    gpkg = GeoPackagePath(testdata_path / "buildings_helsinki.gpkg")
    gpkg_control = GeoPackagePath(
        testdata_path / "buildings_generalized_50k_helsinki.gpkg"
    )

    IntegrationTest(
        input_uri=gpkg.to_input("single_parts"),
        control_uri=gpkg_control.to_input("control"),
        algorithm=GeneralizeBuildings(
            area_threshold_for_all_buildings=5,
            area_threshold_for_low_priority_buildings=100,
            side_threshold=30,
            point_size=15,
            minimum_distance_to_isolated_building=200,
            hole_threshold=75,
            classes_for_low_priority_buildings=[6, 4],
            classes_for_point_buildings=[8],
            classes_for_always_kept_buildings=[],
            unique_key_column="mtk_id",
            building_class_column="kayttotarkoitus",
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
    ).run()


def test_generalize_buildings_100k(testdata_path: Path) -> None:
    gpkg = GeoPackagePath(testdata_path / "buildings_generalized_50k_helsinki.gpkg")
    gpkg_control = GeoPackagePath(
        testdata_path / "buildings_generalized_100k_helsinki.gpkg"
    )

    IntegrationTest(
        input_uri=gpkg.to_input("control"),
        control_uri=gpkg_control.to_input("control"),
        algorithm=GeneralizeBuildings(
            area_threshold_for_all_buildings=10,
            area_threshold_for_low_priority_buildings=500,
            side_threshold=70,
            point_size=30,
            minimum_distance_to_isolated_building=400,
            hole_threshold=150,
            classes_for_low_priority_buildings=[6, 4],
            classes_for_point_buildings=[8],
            classes_for_always_kept_buildings=[],
            unique_key_column="mtk_id",
            building_class_column="kayttotarkoitus",
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
    ).run()


@pytest.mark.parametrize(
    ("input_gdf", "expected_area", "expected_angle"),
    [
        (
            GeoDataFrame(
                geometry=[Polygon([(0, 0), (1, 1), (0, 2), (-1, 1)])], crs="EPSG:3067"
            ),
            2.0,
            135.0,
        ),
        (
            GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:3067"),
            0.0,
            nan,
        ),
        (
            GeoDataFrame(
                {
                    "geometry": [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
                    "original_area": [9999.0],
                    "main_angle": [8888.0],
                },
                crs="EPSG:3067",
            ),
            9999.0,
            8888.0,
        ),
        (
            GeoDataFrame(
                {
                    "geometry": [Point(1, 1)],
                    "original_area": [123.0],
                    "main_angle": [321.0],
                },
                crs="EPSG:3067",
            ),
            123.0,
            321.0,
        ),
    ],
    ids=[
        "polygon without attributes: should compute area and angle",
        "point without attributes",
        "polygon with attributes: should not be changed",
        "Point with attributes: should not be changed",
    ],
)
def test_add_attributes_for_area_and_angle(
    input_gdf: GeoDataFrame, expected_area: float, expected_angle: float
):
    result_gdf = GeneralizeBuildings()._add_attributes_for_area_and_angle(input_gdf)

    assert "original_area" in result_gdf.columns
    assert "main_angle" in result_gdf.columns

    area = result_gdf.loc[0, "original_area"]
    angle = result_gdf.loc[0, "main_angle"]

    assert area == expected_area
    if isna(expected_angle):
        assert isna(angle)
    else:
        assert angle == expected_angle


@pytest.mark.parametrize(
    ("original_area", "building_class", "expected_kept"),
    [
        (40, "normal", False),
        (110, "low", False),
        (150, "low", True),
        (10, "always", True),
        (110, "normal", True),
    ],
    ids=[
        "area below area_threshold_for_all_buildings",
        "low priority building below area_threshold_for_low_priority_buildings",
        "low priority building above area_threshold_for_low_priority_buildings",
        "building to be kept always with small area",
        "normal building above area_threshold_for_all_buildings",
    ],
)
def test_filter_buildings_by_area_and_class(
    original_area: float, building_class: str, expected_kept: bool
):
    input_gdf = GeoDataFrame(
        {
            "original_area": [original_area],
            "class": [building_class],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        crs="EPSG:3067",
    )

    alg = GeneralizeBuildings(
        area_threshold_for_all_buildings=100,
        area_threshold_for_low_priority_buildings=120,
        classes_for_low_priority_buildings=["low"],
        classes_for_always_kept_buildings=["always"],
        building_class_column="class",
        side_threshold=10,
        point_size=10,
        minimum_distance_to_isolated_building=10,
        hole_threshold=10,
        classes_for_point_buildings=[3, 2],
        unique_key_column="id",
    )

    result_gdf = alg._filter_buildings_by_area_and_class(input_gdf)

    assert (len(result_gdf) == 1) == expected_kept


@pytest.mark.parametrize(
    ("original_areas", "classes", "geometries", "expected_count"),
    [
        ([], [], [], 0),
        ([10, 20], ["normal", "low"], [Point(0, 0), Point(1, 1)], 0),
        ([150], ["normal"], [Point(0, 0)], 1),
        (
            [50, 300, 10],
            ["normal", "low", "always"],
            [Point(0, 0), Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Point(1, 1)],
            2,
        ),
        (
            [50, 300, 101],
            ["normal", "low", "normal"],
            [
                Point(0, 0),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            ],
            2,
        ),
    ],
    ids=[
        "empty input GeoDataFrame",
        "two points with areas below area_threshold_for_all_buildings",
        "one normal point above area_threshold_for_all_buildings",
        "one polygon and one point should be retained",
        "both polygons should be retained",
    ],
)
def test_filter_buildings_by_area_and_class_with_varied_geometries(
    original_areas: list[int],
    classes: list[str],
    geometries: list[BaseGeometry],
    expected_count: int,
):
    gdf = GeoDataFrame(
        {
            "original_area": original_areas,
            "class": classes,
            "geometry": geometries,
        },
        crs="EPSG:3857",
    )

    alg = GeneralizeBuildings(
        area_threshold_for_all_buildings=100,
        area_threshold_for_low_priority_buildings=200,
        classes_for_low_priority_buildings=["low"],
        classes_for_always_kept_buildings=["always"],
        building_class_column="class",
        side_threshold=10,
        point_size=10,
        minimum_distance_to_isolated_building=10,
        hole_threshold=10,
        classes_for_point_buildings=[3, 2],
        unique_key_column="id",
    )
    result_gdf = alg._filter_buildings_by_area_and_class(gdf)

    assert len(result_gdf) == expected_count


@pytest.mark.parametrize(
    ("input_gdf", "building_class_column", "unique_key_column", "expected_gdf"),
    [
        (
            GeoDataFrame(
                {"id": [1], "class": ["A"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            ),
            "class",
            "id",
            GeoDataFrame(
                {"id": [1], "class": ["A"], "old_ids": [(1,)]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [2, 1], "class": ["A", "A"]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1.0001, 0), (2, 0), (2, 1), (1.0001, 1)]),
                ],
            ),
            "class",
            "id",
            GeoDataFrame(
                {"id": [1], "class": ["A"], "old_ids": [(2, 1)]},
                geometry=[Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [1, 2], "class": ["A", "B"]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1.0001, 0), (2, 0), (2, 1), (1.0001, 1)]),
                ],
            ),
            "class",
            "id",
            GeoDataFrame(
                {"id": [1, 2], "class": ["A", "B"], "old_ids": [(1,), (2,)]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1.0001, 0), (2, 0), (2, 1), (1.0001, 1)]),
                ],
            ),
        ),
        (
            GeoDataFrame(columns=["id", "class"], geometry=[]),
            "class",
            "id",
            GeoDataFrame(columns=["id", "class"], geometry=[]),
        ),
        (
            GeoDataFrame(
                {"id": [1, 2], "class": ["A", "A"]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
                ],
            ),
            "class",
            "id",
            GeoDataFrame(
                {"id": [1, 2], "class": ["A", "A"], "old_ids": [(1,), (2,)]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
                ],
            ),
        ),
    ],
    ids=[
        "single building",
        "two touching buildings in same class",
        "two touching buildings in different classes",
        "empty GeoDataFrame",
        "two far buildings in same class",
    ],
)
def test_dissolve_touching_buildings(
    input_gdf: GeoDataFrame,
    building_class_column: str,
    unique_key_column: str,
    expected_gdf: GeoDataFrame,
):
    input_gdf = input_gdf.set_index("id")
    expected_gdf = expected_gdf.set_index("id")
    result_gdf = GeneralizeBuildings(
        building_class_column=building_class_column,
    )._dissolve_touching_buildings(
        input_gdf.copy(),
    )

    result_sorted = result_gdf.sort_values(unique_key_column).reset_index(drop=True)
    expected_sorted = expected_gdf.sort_values(unique_key_column).reset_index(drop=True)

    # Check geometries
    for result_geom, expected_geom in zip(
        result_sorted.geometry, expected_sorted.geometry, strict=False
    ):
        assert result_geom.equals(expected_geom), (
            f"Geometries differ: {result_geom} vs {expected_geom}"
        )

    # Check attributes
    result_attrs = result_sorted.drop(columns="geometry")
    expected_attrs = expected_sorted.drop(columns="geometry")
    assert_frame_equal(result_attrs, expected_attrs)


def test_simplify_buildings_calls_cartagen_function_correctly(mocker: "MockerFixture"):
    input_gdf = GeoDataFrame(
        {"id": [1, 2], "class": ["A", "A"], "dissolve_members": [None, None]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ],
    )
    # TODO: this requires not importing the simplify_building function
    # directly, instead requiring to import the buildings module in
    # generalize_buildings.py. Ideally this test should be refactored so that
    # the direct import can be used.
    spy_simplify_building = mocker.spy(buildings, "simplify_building")
    GeneralizeBuildings._simplify_buildings(input_gdf, 3)

    assert spy_simplify_building.call_count == 2


def test_simplify_buildings_raises_on_invalid_geometry():
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
            "Simplify buildings only supports Polygon or MultiPolygon geometries."
        ),
    ):
        GeneralizeBuildings._simplify_buildings(invalid_gdf, 5.0)
