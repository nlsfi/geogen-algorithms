#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_conservation_areas import (
    GeneralizeConservationAreas,
)
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_conservation_areas(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "conservation_areas.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("conservation_areas"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeConservationAreas(
            positive_buffer_coastal_areas=25,
            negative_buffer_coastal_areas=-5,
            positive_buffer_inland_areas=5,
            negative_buffer_inland_areas=-5,
            simplification_tolerance=3,
            area_threshold=1000,
            hole_threshold=2000,
            smoothing=False,
            group_by="layer",
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        reference_uris={
            "water": gpkg.to_input("water_areas"),
        },
        check_missing_reference=False,
    ).run()
