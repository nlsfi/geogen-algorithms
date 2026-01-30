#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import re
from pathlib import Path

import pytest
from geopandas import GeoDataFrame
from pandas.testing import assert_frame_equal
from shapely import Point

from geogenalg.application.generalize_cliffs import GeneralizeCliffs
from geogenalg.core.exceptions import GeometryTypeError, MissingReferenceError
from geogenalg.testing import GeoPackageInput, get_result_and_control
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_cliffs_50k(testdata_path: Path) -> None:
    input_path = testdata_path / "cliffs.gpkg"
    result, control = get_result_and_control(
        GeoPackageInput(input_path, layer_name="cliffs_source"),
        GeoPackageInput(input_path, layer_name="control"),
        GeneralizeCliffs(buffer_size=20, length_threshold=50, reference_key="roads"),
        UNIQUE_ID_COLUMN,
        reference_uris={"roads": GeoPackageInput(input_path, layer_name="roads")},
    )

    assert_frame_equal(result, control)


def test_generalize_cliffs_invalid_geometry_type() -> None:
    with pytest.raises(GeometryTypeError):
        GeneralizeCliffs().execute(
            data=GeoDataFrame(geometry=[Point(0, 0)]), reference_data={}
        )


def test_generalize_cliffs_missing_reference_data(testdata_path: Path) -> None:
    input_path = testdata_path / "cliffs.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="cliffs_source",
    )

    with pytest.raises(
        MissingReferenceError, match=re.escape("Reference data is missing.")
    ):
        GeneralizeCliffs().execute(data=input_data, reference_data={})


def test_generalize_cliffs_invalid_reference_data_geometry_type(
    testdata_path: Path,
) -> None:
    input_path = testdata_path / "cliffs.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="cliffs_source",
    )

    with pytest.raises(GeometryTypeError):
        GeneralizeCliffs().execute(
            data=input_data,
            reference_data={"roads": GeoDataFrame(geometry=[Point(0, 0)])},
        )
