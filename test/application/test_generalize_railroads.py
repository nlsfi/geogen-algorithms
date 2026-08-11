#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_railroads import GeneralizeRailroads
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_railroads(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "railroads.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("railroads"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeRailroads(
            fan_minimum_length=400,
            fan_rail_parallel_distance=6,
            pack_cluster_length_threshold=200,
            pack_track_maximum_length=1000,
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        reference_uris={},
        check_missing_reference=False,
    ).run()
