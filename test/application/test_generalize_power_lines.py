#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_power_lines import GeneralizePowerLines
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_power_lines(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "power_lines.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("power_lines"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizePowerLines(
            class_column="kohdeluokka",
            classes_for_higher_priority_lines=[22311],
            classes_for_merge_parallel_lines=[22311],
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        reference_uris={
            "substations": gpkg.to_input("substations"),
            "fences": gpkg.to_input("fences"),
        },
        check_missing_reference=True,
    ).run()
