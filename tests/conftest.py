#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from geogenalg.application.dissolve_polygons import DissolvePolygons
from geogenalg.application.generalize_building_areas import GeneralizeBuildingAreas
from geogenalg.application.generalize_buildings import GeneralizeBuildings
from geogenalg.application.generalize_cliffs import GeneralizeCliffs
from geogenalg.application.generalize_conservation_areas import (
    GeneralizeConservationAreas,
)
from geogenalg.application.generalize_contours import GeneralizeContours
from geogenalg.application.generalize_fences import GeneralizeFences
from geogenalg.application.generalize_landcover import GeneralizeLandcover
from geogenalg.application.generalize_polygons_to_points import (
    GeneralizePolygonsToPoints,
)
from geogenalg.application.generalize_power_lines import GeneralizePowerLines
from geogenalg.application.generalize_railroads import GeneralizeRailroads
from geogenalg.application.generalize_roads import GeneralizeRoads
from geogenalg.application.generalize_shared_paths import GeneralizeSharedPaths
from geogenalg.application.generalize_slopelines import GeneralizeSlopeLines
from geogenalg.application.generalize_water_areas import GeneralizeWaterAreas
from geogenalg.application.generalize_watercourse_areas import (
    GeneralizeWaterCourseAreas,
)
from geogenalg.testing import GeoPackagePath, TestInputData


@pytest.fixture
def testdata_path() -> Path:
    return Path(__file__).resolve().parent / "testdata"


@pytest.fixture
def algorithm_testdata_path(testdata_path: Path) -> Path:
    return testdata_path / "algo"


@pytest.fixture
def bench_testdata_path(testdata_path: Path) -> Path:
    return testdata_path / "bench"


@pytest.fixture
def dissolve_polygons_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "dissolve_polygons.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("data"),
        control_uri=gpkg.to_input("control"),
        unique_id_column="id",
    )


@pytest.fixture
def dissolve_polygons_algorithm() -> DissolvePolygons:
    return DissolvePolygons(
        hash_prefix="dissolvepolygons",
        by_column=frozenset(),
        inherit_from="most_intersection",
    )


@pytest.fixture
def building_areas_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "building_areas.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("buildings"),
        control_uri=gpkg.to_input("control"),
        unique_id_column="mtk_id",
        reference_uris={
            "parcels": gpkg.to_input("parcels"),
            "roads": gpkg.to_input("roads"),
        },
    )


@pytest.fixture
def building_areas_algorithm() -> GeneralizeBuildingAreas:
    return GeneralizeBuildingAreas(
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
    )


@pytest.fixture
def buildings_50k_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "buildings.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("buildings"),
        control_uri=gpkg.to_input("control_50k"),
        unique_id_column="mtk_id",
    )


@pytest.fixture
def buildings_50k_algorithm() -> GeneralizeBuildings:
    return GeneralizeBuildings(
        point_size=15,
        minimum_distance_to_isolated_building=200,
        hole_threshold=75,
        classes_for_low_priority_buildings=frozenset([6, 4]),
        classes_for_point_buildings=frozenset([8]),
        classes_for_always_kept_buildings=frozenset(),
        building_class_column="kayttotarkoitus",
        main_angle_column="main_angle",
    )


@pytest.fixture
def buildings_100k_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "buildings.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("control_50k"),
        control_uri=gpkg.to_input("control_100k"),
        unique_id_column="mtk_id",
    )


@pytest.fixture
def buildings_100k_algorithm() -> GeneralizeBuildings:
    return GeneralizeBuildings(
        area_threshold_for_all_buildings=10,
        area_threshold_for_low_priority_buildings=500,
        side_threshold=70,
        point_size=30,
        minimum_distance_to_isolated_building=400,
        hole_threshold=150,
        classes_for_low_priority_buildings=frozenset([6, 4]),
        classes_for_point_buildings=frozenset([8]),
        classes_for_always_kept_buildings=frozenset(),
        building_class_column="kayttotarkoitus",
        main_angle_column="main_angle",
    )


@pytest.fixture
def cliffs_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "cliffs.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("cliffs_source"),
        control_uri=gpkg.to_input("control"),
        unique_id_column="mtk_id",
        reference_uris={
            "roads": gpkg.to_input("roads"),
        },
    )


@pytest.fixture
def cliffs_algorithm() -> GeneralizeCliffs:
    return GeneralizeCliffs(
        buffer_size=20.0,
        length_threshold=50.0,
        reference_key="roads",
    )


@pytest.fixture
def conservation_areas_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "conservation_areas.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("conservation_areas"),
        control_uri=gpkg.to_input("control"),
        unique_id_column="mtk_id",
        reference_uris={
            "water_areas": gpkg.to_input("water_areas"),
        },
    )


@pytest.fixture
def conservation_areas_algorithm() -> GeneralizeConservationAreas:
    return GeneralizeConservationAreas(
        positive_buffer_coastal_areas=25,
        negative_buffer_coastal_areas=-5,
        positive_buffer_inland_areas=5,
        negative_buffer_inland_areas=-5,
        simplification_tolerance=3,
        area_threshold=1000,
        hole_threshold=2000,
        smoothing=False,
        group_by=frozenset(["layer"]),
        reference_key="water_areas",
    )


@pytest.fixture
def contours_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "contours.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("contour"),
        control_uri=gpkg.to_input("contour_control"),
        unique_id_column="kmtk_id",
        reference_uris={
            "slope_line": gpkg.to_input("slope_line"),
        },
    )


@pytest.fixture
def contours_algorithm() -> GeneralizeContours:
    return GeneralizeContours(
        interval=5,
        gaussian_filter_strength=8,
        length_threshold=200,
        level_attribute="n60_elevation_value",
        reference_key="slope_line",
    )


@pytest.fixture
def fences_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "fences.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("fences"),
        control_uri=gpkg.to_input("control"),
        unique_id_column="mtk_id",
        reference_uris={
            "masts": gpkg.to_input("masts"),
        },
    )


@pytest.fixture
def fences_algorithm() -> GeneralizeFences:
    return GeneralizeFences(
        closing_fence_area_threshold=2000,
        closing_fence_area_with_mast_threshold=8000,
        fence_length_threshold=80,
        fence_length_threshold_in_closed_area=300,
        simplification_tolerance=4,
        gap_threshold=25,
        attribute_for_line_merge="kohdeluokka",
    )


@pytest.fixture
def landcover_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "landcover.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("marsh"),
        control_uri=gpkg.to_input("control"),
        unique_id_column="mtk_id",
    )


@pytest.fixture
def landcover_algorithm() -> GeneralizeLandcover:
    return GeneralizeLandcover(
        positive_buffer=25,
        negative_buffer=-10,
        simplification_tolerance=15,
        area_threshold=5000,
        hole_threshold=5000,
        smoothing=True,
        buffer_join_style="bevel",
        group_by=frozenset(),
    )


@pytest.fixture
def polygons_to_points_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "polygons_to_points.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("boulders_in_water"),
        control_uri=gpkg.to_input("control"),
        unique_id_column="kmtk_id",
    )


@pytest.fixture
def polygons_to_points_algorithm() -> GeneralizePolygonsToPoints:
    return GeneralizePolygonsToPoints(
        polygon_min_area=1000.0,
    )


@pytest.fixture
def power_lines_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "power_lines.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("power_lines"),
        control_uri=gpkg.to_input("control"),
        unique_id_column="mtk_id",
        reference_uris={
            "substations": gpkg.to_input("substations"),
            "fences": gpkg.to_input("fences"),
        },
    )


@pytest.fixture
def power_lines_algorithm() -> GeneralizePowerLines:
    return GeneralizePowerLines(
        distance_threshold_for_parallel_lines=50.0,
        classes_for_merge_parallel_lines=frozenset([22311]),
        classes_for_higher_priority_lines=frozenset([22311]),
        class_column="kohdeluokka",
        length_threshold=100.0,
        simplification_tolerance=0.0,
        reference_key_fences="fences",
        reference_key_substations="substations",
    )


@pytest.fixture
def railroads_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "railroads.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("railroads"),
        control_uri=gpkg.to_input("control"),
        unique_id_column="mtk_id",
    )


@pytest.fixture
def railroads_algorithm() -> GeneralizeRailroads:
    return GeneralizeRailroads(
        fan_minimum_length=400,
        fan_rail_parallel_distance=6,
        pack_cluster_length_threshold=200,
        pack_track_maximum_length=1000,
    )


@pytest.fixture
def roads_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "roads.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("road_link"),
        control_uri=gpkg.to_input("control"),
        unique_id_column="kmtk_id",
        reference_uris={
            "network": gpkg.to_input("path"),
        },
    )


@pytest.fixture
def roads_algorithm() -> GeneralizeRoads:
    return GeneralizeRoads(
        threshold_distance=10.0,
        threshold_length=75.0,
        reference_key="network",
    )


@pytest.fixture
def shared_paths_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "roads.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("shared_path_link"),
        control_uri=gpkg.to_input("control_shared_paths"),
        unique_id_column="kmtk_id",
        # Use control from GeneralizeRoads test as reference, as these are intended
        # to be used sequentially.
        reference_uris={
            "roads": gpkg.to_input("control"),
        },
    )


@pytest.fixture
def shared_paths_algorithm() -> GeneralizeSharedPaths:
    return GeneralizeSharedPaths(
        detection_distance=25.0,
        minimum_percentage=90.0,
        reference_key="roads",
    )


@pytest.fixture
def slopelines_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "contours.gpkg")
    return TestInputData(
        input_uri=gpkg.to_input("slope_line"),
        control_uri=gpkg.to_input("slope_line_control"),
        unique_id_column="kmtk_id",
        reference_uris={
            "contour_control": gpkg.to_input("contour_control"),
        },
    )


@pytest.fixture
def slopelines_algorithm() -> GeneralizeSlopeLines:
    return GeneralizeSlopeLines(
        tolerance=1.0,
        reference_key="contour_control",
    )


@pytest.fixture
def water_areas_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "water_areas.gpkg")
    return TestInputData(
        input_uri=[
            gpkg.to_input("areas"),
            gpkg.to_input("shoreline"),
        ],
        control_uri=gpkg.to_input("control"),
        unique_id_column="kmtk_id",
    )


@pytest.fixture
def water_areas_algorithm() -> GeneralizeWaterAreas:
    return GeneralizeWaterAreas(
        min_area=4000.0,
        area_simplification_tolerance=10.0,
        thin_section_width=20.0,
        thin_section_min_size=200.0,
        thin_section_exaggerate_by=3.0,
        island_min_area=100.0,
        island_min_width=185.0,
        island_min_elongation=0.25,
        island_exaggerate_by=3.0,
        island_simplification_tolerance=10.0,
        smoothing_passes=3,
        preserve_shoreline_sections_column="shoreline_type_id",
        preserve_shoreline_sections_values=frozenset([3]),
    )


@pytest.fixture
def watercourse_areas_input(algorithm_testdata_path: Path) -> TestInputData:
    gpkg = GeoPackagePath(algorithm_testdata_path / "watercourse_areas.gpkg")
    return TestInputData(
        input_uri=[
            gpkg.to_input("watercourse_part_area"),
            gpkg.to_input("shoreline"),
        ],
        control_uri=gpkg.to_input("control"),
        unique_id_column="kmtk_id",
    )


@pytest.fixture
def watercourse_areas_algorithm() -> GeneralizeWaterCourseAreas:
    return GeneralizeWaterCourseAreas(
        min_area=4000.0,
        area_simplification_tolerance=10.0,
        thin_section_width=20.0,
        thin_section_min_size=200.0,
        thin_section_exaggerate_by=0.0,
        island_min_area=100.0,
        island_min_width=185.0,
        island_min_elongation=0.25,
        island_exaggerate_by=3.0,
        island_simplification_tolerance=10.0,
        smoothing_passes=3,
        line_transform_width=30.0,
        line_min_length=200.0,
        min_new_section_length=200.0,
        width_check_distance=10.0,
        feature_type_column="feature_type",
    )
