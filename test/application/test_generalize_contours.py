#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_contours import GeneralizeContours
from geogenalg.testing import GeoPackagePath
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_contours(
    testdata_path: Path,
    tmp_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "contours_2.gpkg")

    slope_data = read_gdf_from_file_and_set_index(
        testdata_path / "contours_2.gpkg",
        UNIQUE_ID_COLUMN,
        layer="symbol",
    )
    slope_data = slope_data[slope_data["symbol_type_id"] == 52192]

    slope_path = tmp_path / "slope.gpkg"
    slope_data.to_file(slope_path, layer="slope")

    IntegrationTest(
        input_uri=gpkg.to_input("contour"),
        control_uri=gpkg.to_input("contour_control"),
        algorithm=GeneralizeContours(
            interval=5,
            gaussian_filter_strength=8,
            length_threshold=200,
            level_attribute="n60_elevation_value",
            reference_key="slope",
        ),
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=True,
        reference_uris={
            "slope": GeoPackagePath(slope_path).to_input("slope"),
        },
        assert_function_arguments={
            "check_less_precise": True,
        },
        dummy_data_mandatory_columns=["n60_elevation_value"],
    ).run()
