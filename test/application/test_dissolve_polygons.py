#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.dissolve_polygons import DissolvePolygons
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "id"


def test_dissolve_polygons(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "dissolve_polygons.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("data"),
        control_uri=gpkg.to_input("control"),
        algorithm=DissolvePolygons(),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
    ).run()
