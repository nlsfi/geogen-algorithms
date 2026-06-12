#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_contours import GeneralizeContours
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_contours(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "contours.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input("contour"),
        control_uri=gpkg.to_input("contour_control"),
        algorithm=GeneralizeContours(
            interval=5,
            gaussian_filter_strength=8,
            length_threshold=200,
            level_attribute="n60_elevation_value",
            reference_key="slope_line",
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
        reference_uris={
            "slope_line": gpkg.to_input("slope_line"),
        },
        assert_function_arguments={
            "check_less_precise": True,
        },
        dummy_data_mandatory_columns=["n60_elevation_value"],
    ).run()
