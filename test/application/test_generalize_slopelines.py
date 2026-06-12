#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_slopelines import GeneralizeSlopeLines
from geogenalg.testing import GeoPackagePath
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_slopelines(
    testdata_path: Path,
    tmp_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "contours.gpkg")

    contour_control = read_gdf_from_file_and_set_index(
        testdata_path / "contours.gpkg",
        UNIQUE_ID_COLUMN,
        layer="contour_control",
    )

    reference_path = tmp_path / "contour_control.gpkg"
    contour_control.to_file(reference_path, layer="contour_control")

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
            "contour_control": GeoPackagePath(reference_path).to_input(
                "contour_control"
            ),
        },
        assert_function_arguments={
            "check_less_precise": False,
        },
    ).run()
