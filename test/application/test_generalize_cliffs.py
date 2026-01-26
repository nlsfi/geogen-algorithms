#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_cliffs import GeneralizeCliffs
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_cliffs(testdata_path: Path) -> None:
    gpkg = GeoPackagePath(testdata_path / "cliffs.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("cliffs_source"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeCliffs(),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=True,
        reference_uris={
            "roads": gpkg.to_input("roads"),
        },
    ).run()
