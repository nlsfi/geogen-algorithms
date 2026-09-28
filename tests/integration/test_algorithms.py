#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from pathlib import Path

import pytest

from geogenalg.application.generalize_points import GeneralizePoints
from geogenalg.application.keep_intersection import KeepIntersection
from geogenalg.application.remove_overlap import RemoveOverlap
from geogenalg.testing import AlgorithmTestInput, GeoPackagePath
from tests.integration.runner import ExpectedResultColumns, IntegrationTest


def test_dissolve_polygons(dissolve_polygons_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=dissolve_polygons_input,
        check_missing_reference=False,
    ).run()


def test_generalize_building_areas(building_areas_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=building_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["building_function_id"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="none",
        ),
    ).run()


def test_generalize_buildings_50k(buildings_50k_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=buildings_50k_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["kayttotarkoitus"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
            mandatory_extra_columns=frozenset(["main_angle"]),
        ),
    ).run()


def test_generalize_buildings_100k(buildings_100k_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=buildings_100k_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["kayttotarkoitus"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
            mandatory_extra_columns=frozenset(["main_angle"]),
        ),
    ).run()


def test_generalize_cliffs(cliffs_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=cliffs_input,
        check_missing_reference=True,
    ).run()


def test_generalize_conservation_areas(
    conservation_areas_input: AlgorithmTestInput,
) -> None:
    IntegrationTest(
        algorithm_input=conservation_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["layer"]),
    ).run()


def test_generalize_contours(contours_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=contours_input,
        check_missing_reference=False,
        assert_function_arguments={
            "check_less_precise": True,
        },
        dummy_data_mandatory_columns=frozenset(["n60_elevation_value"]),
    ).run()


def test_generalize_fences(fences_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=fences_input,
        check_missing_reference=True,
        dummy_data_mandatory_columns=frozenset(["kohdeluokka"]),
    ).run()


def test_generalize_landcover(landcover_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=landcover_input,
        check_missing_reference=False,
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
    testdata_path: Path,
    input_layer: str,
    control_layer: str,
    algorithm: GeneralizePoints,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "points.gpkg")

    IntegrationTest(
        algorithm_input=AlgorithmTestInput(
            input_uri=gpkg.to_input(input_layer),
            control_uri=gpkg.to_input(control_layer),
            algorithm=algorithm,
            unique_id_column="kmtk_id",
        ),
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["boulder_in_water_type_id"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
            mandatory_extra_columns=frozenset(["is_cluster"]),
        ),
    ).run()


def test_generalize_polygons_to_points(
    polygons_to_points_input: AlgorithmTestInput,
) -> None:
    IntegrationTest(
        algorithm_input=polygons_to_points_input,
        check_missing_reference=False,
    ).run()


def test_generalize_power_lines(power_lines_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=power_lines_input,
        check_missing_reference=True,
        dummy_data_mandatory_columns=frozenset(["kohdeluokka"]),
    ).run()


def test_generalize_railroads(railroads_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=railroads_input,
        check_missing_reference=False,
    ).run()


def test_generalize_roads(roads_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=roads_input,
        check_missing_reference=False,
    ).run()


def test_generalize_shared_paths(shared_paths_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=shared_paths_input,
        check_missing_reference=True,
    ).run()


def test_generalize_slopelines(slopelines_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=slopelines_input,
        check_missing_reference=True,
        assert_function_arguments={
            "check_less_precise": False,
        },
    ).run()


def test_generalize_water_areas(water_areas_input: AlgorithmTestInput):
    IntegrationTest(
        algorithm_input=water_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["shoreline_type_id"]),
        expected_result_columns=ExpectedResultColumns(inherit="input"),
    ).run()


def test_generalize_watercourse_areas(watercourse_areas_input: AlgorithmTestInput):
    IntegrationTest(
        algorithm_input=watercourse_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["shoreline_type_id"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
            mandatory_extra_columns=frozenset(["feature_type"]),
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
def test_keep_intersection(
    testdata_path: Path,
    layer_suffix: str,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "keep_intersection.gpkg")

    IntegrationTest(
        algorithm_input=AlgorithmTestInput(
            input_uri=gpkg.to_input(f"data_{layer_suffix}"),
            algorithm=KeepIntersection(
                reference_key="mask",
            ),
            unique_id_column="uuid",
            reference_uris={
                "mask": gpkg.to_input("mask"),
            },
            control_uri=gpkg.to_input(f"control_{layer_suffix}"),
        ),
        check_missing_reference=True,
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
    testdata_path: Path,
    layer_suffix: str,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "remove_overlap.gpkg")

    IntegrationTest(
        algorithm_input=AlgorithmTestInput(
            input_uri=gpkg.to_input(f"data_{layer_suffix}"),
            control_uri=gpkg.to_input(f"control_{layer_suffix}"),
            algorithm=RemoveOverlap(
                reference_key="mask",
            ),
            unique_id_column="uuid",
            reference_uris={
                "mask": gpkg.to_input("mask"),
            },
        ),
        check_missing_reference=True,
    ).run()
