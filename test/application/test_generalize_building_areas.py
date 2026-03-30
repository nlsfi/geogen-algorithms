#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_building_areas import GeneralizeBuildingAreas
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_building_areas(testdata_path: Path) -> None:
    gpkg = GeoPackagePath(testdata_path / "building_areas.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("buildings"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeBuildingAreas(
            parcel_building_size_threshold=4000.0,
            parcel_coverage_threshold=5.0,
            parcel_buffer_distance=20.0,
            parcel_building_type_column="building_function_id",
            parcel_building_type_to_select=1,
            geometry_buildings_simplify_tolerance=10.0,
            roads_buffer_distance=10.0,
            threshold_building_area_far=20000.0,
            threshold_building_area_near=4000.0,
            near_area_distance=50.0,
            reference_key_parcels="parcels",
            reference_key_roads="roads",
            positive_buffer=10,
            negative_buffer=-10.0,
            simplification_tolerance=4.0,
            hole_threshold=7500,
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        reference_uris={
            "parcels": gpkg.to_input("parcels"),
            "roads": gpkg.to_input("roads"),
        },
        check_missing_reference=True,
    ).run()
