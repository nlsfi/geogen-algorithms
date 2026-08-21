#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from pathlib import Path

from conftest import ExpectedResultColumns, IntegrationTest

from geogenalg.application.generalize_slopelines import GeneralizeSlopeLines
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_slopelines(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "contours.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("slope_line"),
        control_uri=gpkg.to_input("slope_line_control"),
        algorithm=GeneralizeSlopeLines(
            tolerance=1.0,
            reference_key="contour_control",
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=True,
        reference_uris={
            "contour_control": gpkg.to_input("contour_control"),
        },
        assert_function_arguments={
            "check_less_precise": False,
        },
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
        ),
    ).run()
