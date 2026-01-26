#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_roads import GeneralizeRoads
from geogenalg.testing import (
    GeoPackagePath,
)

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_roads(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "roads.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("road_link"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeRoads(
            threshold_distance=10.0,
            threshold_length=75.0,
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        reference_uris={
            "path": gpkg.to_input("path"),
        },
        check_missing_reference=False,
    ).run()
