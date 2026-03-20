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
        algorithm=GeneralizeWaterCourseAreas(
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
            reference_key="shoreline",
            line_transform_width=30.0,
            line_min_length=200.0,
            min_new_section_length=200.0,
            width_check_distance=10.0,
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
    ).run()
