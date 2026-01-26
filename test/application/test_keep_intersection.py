#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from conftest import IntegrationTest

from geogenalg.application.keep_intersection import KeepIntersection
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "uuid"


@pytest.mark.parametrize(
    ("layer_suffix"),
    [
        ("polygons"),
        ("lines"),
        ("points"),
    ],
    ids=[
        "polygons",
        "lines",
        "points",
    ],
)
def test_keep_intersection(
    testdata_path: Path,
    layer_suffix: str,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "keep_intersection.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input(f"data_{layer_suffix}"),
        control_uri=gpkg.to_input(f"control_{layer_suffix}"),
        algorithm=KeepIntersection(),
        unique_id_column=UNIQUE_ID_COLUMN,
        reference_uris={
            "mask": gpkg.to_input("mask"),
        },
        check_missing_reference=True,
    ).run()
