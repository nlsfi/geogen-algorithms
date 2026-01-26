#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from conftest import IntegrationTest

from geogenalg.application.generalize_roads import GeneralizeRoads
from geogenalg.testing import (
    GeoPackagePath,
)

UNIQUE_ID_COLUMN = "kmtk_id"


@pytest.mark.parametrize(
    (
        "gpkg_file",
        "input_layer",
    ),
    [
        ("roads.gpkg", "road_link"),
    ],
    ids=[
        "roads_to_generalize",
    ],
)
def test_generalize_roads(
    testdata_path: Path,
    gpkg_file: str,
    input_layer: str,
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
