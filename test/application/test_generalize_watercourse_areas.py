#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_watercourse_areas import (
    GeneralizeWaterCourseAreas,
)
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_watercourse_areas(testdata_path: Path):
    gpkg = GeoPackagePath(testdata_path / "watercourse_areas.gpkg")
    IntegrationTest(
        input_uri=gpkg.to_input("watercourse_part_area"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeWaterCourseAreas(),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
    ).run()
