#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from geogenalg.application import BaseAlgorithm
from geogenalg.application.generalize_points import GeneralizePoints
from geogenalg.application.keep_intersection import KeepIntersection
from geogenalg.application.remove_overlap import RemoveOverlap
from geogenalg.testing import GeoPackagePath, TestInputData
from tests.bench.algo.runner import AlgorithmBenchmark


@pytest.mark.benchmark
def test_dissolve_polygons(
    benchmark: BenchmarkFixture,
    dissolve_polygons_input: TestInputData,
    dissolve_polygons_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=dissolve_polygons_input,
        algorithm=dissolve_polygons_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_building_areas(
    benchmark: BenchmarkFixture,
    building_areas_input: TestInputData,
    building_areas_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=building_areas_input,
        algorithm=building_areas_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_buildings_50k(
    benchmark: BenchmarkFixture,
    buildings_50k_input: TestInputData,
    buildings_50k_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=buildings_50k_input,
        algorithm=buildings_50k_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_buildings_100k(
    benchmark: BenchmarkFixture,
    buildings_100k_input: TestInputData,
    buildings_100k_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=buildings_100k_input,
        algorithm=buildings_100k_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_cliffs(
    benchmark: BenchmarkFixture,
    cliffs_input: TestInputData,
    cliffs_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=cliffs_input,
        algorithm=cliffs_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_conservation_areas(
    benchmark: BenchmarkFixture,
    conservation_areas_input: TestInputData,
    conservation_areas_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=conservation_areas_input,
        algorithm=conservation_areas_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_contours(
    benchmark: BenchmarkFixture,
    contours_input: TestInputData,
    contours_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=contours_input,
        algorithm=contours_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_fences(
    benchmark: BenchmarkFixture,
    fences_input: TestInputData,
    fences_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=fences_input,
        algorithm=fences_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_landcover(
    benchmark: BenchmarkFixture,
    landcover_input: TestInputData,
    landcover_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=landcover_input,
        algorithm=landcover_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_polygons_to_points(
    benchmark: BenchmarkFixture,
    polygons_to_points_input: TestInputData,
    polygons_to_points_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=polygons_to_points_input,
        algorithm=polygons_to_points_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_power_lines(
    benchmark: BenchmarkFixture,
    power_lines_input: TestInputData,
    power_lines_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=power_lines_input,
        algorithm=power_lines_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_railroads(
    benchmark: BenchmarkFixture,
    railroads_input: TestInputData,
    railroads_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=railroads_input,
        algorithm=railroads_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_roads(
    benchmark: BenchmarkFixture,
    roads_input: TestInputData,
    roads_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=roads_input,
        algorithm=roads_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_shared_paths(
    benchmark: BenchmarkFixture,
    shared_paths_input: TestInputData,
    shared_paths_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=shared_paths_input,
        algorithm=shared_paths_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_slopelines(
    benchmark: BenchmarkFixture,
    slopelines_input: TestInputData,
    slopelines_algorithm: BaseAlgorithm,
) -> None:
    AlgorithmBenchmark(
        input_data=slopelines_input,
        algorithm=slopelines_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_water_areas(
    benchmark: BenchmarkFixture,
    water_areas_input: TestInputData,
    water_areas_algorithm: BaseAlgorithm,
):
    AlgorithmBenchmark(
        input_data=water_areas_input,
        algorithm=water_areas_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
def test_generalize_watercourse_areas(
    benchmark: BenchmarkFixture,
    watercourse_areas_input: TestInputData,
    watercourse_areas_algorithm: BaseAlgorithm,
):
    AlgorithmBenchmark(
        input_data=watercourse_areas_input,
        algorithm=watercourse_areas_algorithm,
    ).run(benchmark)


@pytest.mark.benchmark
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
    benchmark: BenchmarkFixture,
    algorithm_testdata_path: Path,
    layer_suffix: str,
) -> None:
    gpkg = GeoPackagePath(algorithm_testdata_path / "keep_intersection.gpkg")
    AlgorithmBenchmark(
        input_data=TestInputData(
            input_uri=gpkg.to_input(f"data_{layer_suffix}"),
            unique_id_column="uuid",
            reference_uris={
                "mask": gpkg.to_input("mask"),
            },
            control_uri=gpkg.to_input(f"control_{layer_suffix}"),
        ),
        algorithm=KeepIntersection(
            reference_key="mask",
        ),
    ).run(benchmark)


@pytest.mark.benchmark
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
    benchmark: BenchmarkFixture,
    algorithm_testdata_path: Path,
    layer_suffix: str,
) -> None:
    gpkg = GeoPackagePath(algorithm_testdata_path / "remove_overlap.gpkg")
    AlgorithmBenchmark(
        input_data=TestInputData(
            input_uri=gpkg.to_input(f"data_{layer_suffix}"),
            control_uri=gpkg.to_input(f"control_{layer_suffix}"),
            unique_id_column="uuid",
            reference_uris={
                "mask": gpkg.to_input("mask"),
            },
        ),
        algorithm=RemoveOverlap(
            reference_key="mask",
        ),
    ).run(benchmark)


@pytest.mark.benchmark
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
    benchmark: BenchmarkFixture,
    algorithm_testdata_path: Path,
    input_layer: str,
    control_layer: str,
    algorithm: GeneralizePoints,
) -> None:
    gpkg = GeoPackagePath(algorithm_testdata_path / "points.gpkg")
    AlgorithmBenchmark(
        input_data=TestInputData(
            input_uri=gpkg.to_input(input_layer),
            control_uri=gpkg.to_input(control_layer),
            unique_id_column="kmtk_id",
        ),
        algorithm=algorithm,
    ).run(benchmark)
