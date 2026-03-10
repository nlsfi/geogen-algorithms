#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application import supports_identity
from geogenalg.application.generalize_building_areas_by_parcel import (
    GeneralizeBuildingAreasByParcel,
)
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_building_areas_by_parcel(testdata_path: Path) -> None:
    gpkg = GeoPackagePath(testdata_path / "buildings_helsinki.gpkg")
    gpkg_reference = GeoPackagePath(testdata_path / "properties_helsinki.gpkg")

    # TODO: this is a hacky way to get around testing algorithm which without
    # identity support produces unpredictable indexes. Fix index handling
    # in GeneralizeBuildingAreasByParcel and add a better way to handle such algorithms.
    supports_identity(GeneralizeBuildingAreasByParcel)

    IntegrationTest(
        input_uri=gpkg.to_input("single_parts"),
        control_uri=gpkg_reference.to_input("control"),
        algorithm=GeneralizeBuildingAreasByParcel(
            building_area_threshold=4000,
            coverage_threshold=5,
            buffer_distance=20,
            building_type_column="kohdeluokka",
            building_type_to_select=42211,
            building_coverage_column="building_coverage",
            reference_key="parcels",
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        reference_uris={
            "parcels": gpkg_reference.to_input("property_borders"),
        },
        check_missing_reference=True,
    ).run()
