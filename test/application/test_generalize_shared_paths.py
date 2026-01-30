#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_shared_paths import GeneralizeSharedPaths
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_shared_paths(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "roads.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("shared_path_link"),
        control_uri=gpkg.to_input("control_shared_paths"),
        algorithm=GeneralizeSharedPaths(
            detection_distance=25.0,
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        # Use control from GeneralizeRoads test as reference, as these are intended
        # to be used sequentially.
        reference_uris={
            "roads": gpkg.to_input("control"),
        },
        check_missing_reference=True,
    ).run()
