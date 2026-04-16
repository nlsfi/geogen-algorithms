#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_building_areas_by_parcel import (
    GeneralizeBuildingAreasByParcel,
)
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_building_areas_by_parcel(testdata_path: Path) -> None:
    gpkg = GeoPackagePath(testdata_path / "building_areas_by_parcel.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("buildings"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeBuildingAreasByParcel(
            building_area_threshold=4000,
            coverage_threshold=5,
            buffer_distance=20,
            building_type_column="kohdeluokka",
            building_type_to_select=42211,
            reference_key="parcels",
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        reference_uris={
            "parcels": gpkg.to_input("property_borders"),
        },
        check_missing_reference=True,
        dummy_data_mandatory_columns=["kohdeluokka"],
    ).run()
