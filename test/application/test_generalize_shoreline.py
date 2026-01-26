#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_shoreline import GeneralizeShoreline
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_shoreline(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "lakes_to_shoreline.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("original_shoreline"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeShoreline(),
        unique_id_column=UNIQUE_ID_COLUMN,
        reference_uris={
            "areas": gpkg.to_input("generalized_lakes"),
        },
        check_missing_reference=True,
    ).run()
