#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_building_areas_by_geometry import (
    GeneralizeBuildingAreasByGeometry,
)
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_building_areas_by_geometry(testdata_path: Path) -> None:
    gpkg = GeoPackagePath(testdata_path / "building_areas_by_geometry.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("buildings"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeBuildingAreasByGeometry(
            buildings_simplify_tolerance=10.0,
            boffet_area_buffer=10.0,
            boffet_area_erosion=10.0,
        ),
        reference_uris={
            "roads": gpkg.to_input("roads"),
        },
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
    ).run()
