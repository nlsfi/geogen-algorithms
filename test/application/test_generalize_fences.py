#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_fences import GeneralizeFences
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_fences(testdata_path: Path) -> None:
    gpkg = GeoPackagePath(testdata_path / "fences_rovaniemi.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("mtk_fences"),
        control_uri=gpkg.to_input("generalized_fences"),
        algorithm=GeneralizeFences(
            closing_fence_area_threshold=2000,
            closing_fence_area_with_mast_threshold=8000,
            fence_length_threshold=80,
            fence_length_threshold_in_closed_area=300,
            simplification_tolerance=4,
            gap_threshold=25,
            attribute_for_line_merge="kohdeluokka",
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=True,
        reference_uris={
            "masts": gpkg.to_input("mtk_masts"),
        },
    ).run()
