#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_water_areas import GeneralizeWaterAreas
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_lakes(testdata_path: Path):
    gpkg = GeoPackagePath(testdata_path / "lakes.gpkg")
    IntegrationTest(
        input_uri=gpkg.to_input("lake_part"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeWaterAreas(
            min_area=4000.0,
            island_min_area=100.0,
            area_simplification_tolerance=10.0,
            thin_section_width=20.0,
            thin_section_min_size=200.0,
            thin_section_exaggerate_by=3.0,
            island_min_width=185.0,
            island_min_elongation=0.25,
            island_exaggerate_by=3.0,
            island_simplification_tolerance=10.0,
            smoothing_passes=3,
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
    ).run()


def test_generalize_sea(testdata_path: Path):
    gpkg = GeoPackagePath(testdata_path / "sea.gpkg")
    IntegrationTest(
        input_uri=gpkg.to_input("sea_part"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeWaterAreas(
            min_area=4000.0,
            island_min_area=100.0,
            area_simplification_tolerance=10.0,
            thin_section_width=20.0,
            thin_section_min_size=200.0,
            thin_section_exaggerate_by=3.0,
            island_min_width=185.0,
            island_min_elongation=0.25,
            island_exaggerate_by=3.0,
            island_simplification_tolerance=10.0,
            smoothing_passes=3,
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        reference_uris={
            "shoreline": gpkg.to_input("shoreline"),
        },
        check_missing_reference=False,
    ).run()
