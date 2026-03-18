#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_polygons_to_points import (
    GeneralizePolygonsToPoints,
)
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_polygons_to_points(testdata_path: Path) -> None:
    gpkg = GeoPackagePath(testdata_path / "polygons_to_points.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("boulders_in_water"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizePolygonsToPoints(polygon_min_area=1000.0),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
    ).run()
