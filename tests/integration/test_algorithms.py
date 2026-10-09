#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from pathlib import Path

import pytest

from geogenalg.application import BaseAlgorithm
from geogenalg.application.generalize_building_areas import GeneralizeBuildingAreas
from geogenalg.application.generalize_points import GeneralizePoints
from geogenalg.application.keep_intersection import KeepIntersection
from geogenalg.application.remove_overlap import RemoveOverlap
from geogenalg.testing import GeoPackagePath, TestInputData
from tests.integration.runner import ExpectedResultColumns, IntegrationTest


def test_dissolve_polygons(
    dissolve_polygons_input: TestInputData,
    dissolve_polygons_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=dissolve_polygons_input,
        check_missing_reference=False,
        algorithm=dissolve_polygons_algorithm,
    ).run()


def test_generalize_building_areas(
    building_areas_input: TestInputData,
    building_areas_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=building_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["building_function_id"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="specific",
            specific_columns=frozenset(["building_area_type"]),
        ),
        algorithm=building_areas_algorithm,
    ).run()


def test_generalize_buildings_50k(
    buildings_50k_input: TestInputData,
    buildings_50k_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=buildings_50k_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["kayttotarkoitus"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
            mandatory_extra_columns=frozenset(["main_angle"]),
        ),
        algorithm=buildings_50k_algorithm,
    ).run()


def test_generalize_buildings_100k(
    buildings_100k_input: TestInputData,
    buildings_100k_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=buildings_100k_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["kayttotarkoitus"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
            mandatory_extra_columns=frozenset(["main_angle"]),
        ),
        algorithm=buildings_100k_algorithm,
    ).run()


def test_generalize_cliffs(
    cliffs_input: TestInputData,
    cliffs_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=cliffs_input,
        check_missing_reference=True,
        algorithm=cliffs_algorithm,
    ).run()


def test_generalize_conservation_areas(
    conservation_areas_input: TestInputData,
    conservation_areas_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=conservation_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["layer"]),
        algorithm=conservation_areas_algorithm,
    ).run()


def test_generalize_contours(
    contours_input: TestInputData,
    contours_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=contours_input,
        check_missing_reference=False,
        assert_function_arguments={
            "check_less_precise": True,
        },
        dummy_data_mandatory_columns=frozenset(["n60_elevation_value"]),
        algorithm=contours_algorithm,
    ).run()


def test_generalize_fences(
    fences_input: TestInputData,
    fences_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=fences_input,
        check_missing_reference=True,
        dummy_data_mandatory_columns=frozenset(["kohdeluokka"]),
        algorithm=fences_algorithm,
    ).run()


def test_generalize_landcover(
    landcover_input: TestInputData,
    landcover_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=landcover_input,
        check_missing_reference=False,
        algorithm=landcover_algorithm,
    ).run()


def test_generalize_polygons_to_points(
    polygons_to_points_input: TestInputData,
    polygons_to_points_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=polygons_to_points_input,
        check_missing_reference=False,
        algorithm=polygons_to_points_algorithm,
    ).run()


def test_generalize_power_lines(
    power_lines_input: TestInputData,
    power_lines_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=power_lines_input,
        check_missing_reference=True,
        dummy_data_mandatory_columns=frozenset(["kohdeluokka"]),
        algorithm=power_lines_algorithm,
    ).run()


def test_generalize_railroads(
    railroads_input: TestInputData,
    railroads_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=railroads_input,
        check_missing_reference=False,
        algorithm=railroads_algorithm,
    ).run()


def test_generalize_roads(
    roads_input: TestInputData,
    roads_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=roads_input,
        check_missing_reference=False,
        algorithm=roads_algorithm,
    ).run()


def test_generalize_shared_paths(
    shared_paths_input: TestInputData,
    shared_paths_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=shared_paths_input,
        check_missing_reference=True,
        algorithm=shared_paths_algorithm,
    ).run()


def test_generalize_slopelines(
    slopelines_input: TestInputData,
    slopelines_algorithm: BaseAlgorithm,
) -> None:
    IntegrationTest(
        input_data=slopelines_input,
        check_missing_reference=True,
        assert_function_arguments={
            "check_less_precise": False,
        },
        algorithm=slopelines_algorithm,
    ).run()


def test_generalize_water_areas(
    water_areas_input: TestInputData,
    water_areas_algorithm: BaseAlgorithm,
):
    IntegrationTest(
        input_data=water_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["shoreline_type_id"]),
        expected_result_columns=ExpectedResultColumns(inherit="input"),
        algorithm=water_areas_algorithm,
    ).run()


def test_generalize_watercourse_areas(
    watercourse_areas_input: TestInputData,
    watercourse_areas_algorithm: BaseAlgorithm,
):
    IntegrationTest(
        input_data=watercourse_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["shoreline_type_id"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
            mandatory_extra_columns=frozenset(["feature_type"]),
        ),
        algorithm=watercourse_areas_algorithm,
    ).run()


def test_generalize_watercourse_lines(
    watercourse_lines_input: TestInputData,
    watercourse_lines_algorithm: BaseAlgorithm,
):
    IntegrationTest(
        input_data=watercourse_lines_input,
        check_missing_reference=False,
        algorithm=watercourse_lines_algorithm,
        dummy_data_mandatory_columns=frozenset(["watercourse_line_width_category_id"]),
    ).run()


@pytest.mark.parametrize(
    ("layer_suffix"),
    [
        ("polygons"),
        ("lines"),
        ("points"),
    ],
    ids=[
        "polygons",
        "lines",
        "points",
    ],
)
def test_keep_intersection(
    algorithm_testdata_path: Path,
    layer_suffix: str,
) -> None:
    gpkg = GeoPackagePath(algorithm_testdata_path / "keep_intersection.gpkg")

    IntegrationTest(
        input_data=TestInputData(
            input_uri=gpkg.to_input(f"data_{layer_suffix}"),
            unique_id_column="uuid",
            reference_uris={
                "mask": gpkg.to_input("mask"),
            },
            control_uri=gpkg.to_input(f"control_{layer_suffix}"),
        ),
        check_missing_reference=True,
        algorithm=KeepIntersection(
            reference_key="mask",
        ),
    ).run()


@pytest.mark.parametrize(
    ("layer_suffix"),
    [
        ("polygons"),
        ("lines"),
        ("points"),
    ],
    ids=[
        "polygons",
        "lines",
        "points",
    ],
)
def test_remove_overlap(
    algorithm_testdata_path: Path,
    layer_suffix: str,
) -> None:
    gpkg = GeoPackagePath(algorithm_testdata_path / "remove_overlap.gpkg")

    IntegrationTest(
        input_data=TestInputData(
            input_uri=gpkg.to_input(f"data_{layer_suffix}"),
            control_uri=gpkg.to_input(f"control_{layer_suffix}"),
            unique_id_column="uuid",
            reference_uris={
                "mask": gpkg.to_input("mask"),
            },
        ),
        check_missing_reference=True,
        algorithm=RemoveOverlap(
            reference_key="mask",
        ),
    ).run()


@pytest.mark.parametrize(
    (
        "input_layer",
        "control_layer",
        "algorithm",
    ),
    [
        (
            "boulder_in_water",
            "control_no_displacement",
            GeneralizePoints(
                cluster_distance=30.0,
                displace=False,
                displace_threshold=70.0,
                displace_points_iterations=10,
                aggregation_functions=None,
                is_cluster_column="is_cluster",
            ),
        ),
        (
            "boulder_in_water",
            "control_aggfunc",
            GeneralizePoints(
                cluster_distance=30.0,
                displace=False,
                displace_threshold=70.0,
                displace_points_iterations=10,
                aggregation_functions={
                    "boulder_in_water_type_id": lambda values: min(values)
                    if len(values) == set(values)
                    else 2
                },
                is_cluster_column="is_cluster",
            ),
        ),
        (
            "boulder_in_water",
            "control_displacement",
            GeneralizePoints(
                cluster_distance=30.0,
                displace=True,
                displace_threshold=70.0,
                displace_points_iterations=10,
                aggregation_functions=None,
                is_cluster_column="is_cluster",
            ),
        ),
    ],
    ids=[
        "points",
        "aggfunc",
        "displacement",
    ],
)
def test_generalize_points(
    algorithm_testdata_path: Path,
    input_layer: str,
    control_layer: str,
    algorithm: GeneralizePoints,
) -> None:
    gpkg = GeoPackagePath(algorithm_testdata_path / "points.gpkg")

    IntegrationTest(
        input_data=TestInputData(
            input_uri=gpkg.to_input(input_layer),
            control_uri=gpkg.to_input(control_layer),
            unique_id_column="kmtk_id",
        ),
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["boulder_in_water_type_id"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
            mandatory_extra_columns=frozenset(["is_cluster"]),
        ),
        algorithm=algorithm,
    ).run()


def test_generalize_tall_building_areas(
    tall_building_areas_input: TestInputData,
) -> None:
    IntegrationTest(
        input_data=tall_building_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(
            ["building_function_id", "kerrosluku"],
        ),
        expected_result_columns=ExpectedResultColumns(
            inherit="specific",
            specific_columns=frozenset(["building_area_type"]),
        ),
        algorithm=GeneralizeBuildingAreas(
            building_size_filter_threshold=4000.0,
            parcel_coverage_threshold=5.0,
            parcel_buffer_distance=20.0,
            building_filter_column="building_function_id",
            classes_for_filtering=frozenset([1]),
            buildings_simplify_tolerance=10.0,
            roads_buffer_distance=10.0,
            threshold_building_area_far=20000.0,
            threshold_building_area_near=4000.0,
            near_area_distance=50.0,
            reference_key_parcels="parcels",
            reference_key_roads="roads",
            positive_buffer=10.0,
            negative_buffer=-10.0,
            simplification_tolerance=4.0,
            hole_threshold=7500,
            height_class_column="kerrosluku",
            tall_building_classes=frozenset([2]),
            sliver_erosion_distance=5,
        ),
    ).run()
