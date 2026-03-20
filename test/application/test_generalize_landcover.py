#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_landcover import GeneralizeLandcover
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_landcover(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "landcover.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("marsh"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeLandcover(
            positive_buffer=25,
            negative_buffer=-10,
            simplification_tolerance=15,
            area_threshold=5000,
            hole_threshold=5000,
            smoothing=True,
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
    ).run()
