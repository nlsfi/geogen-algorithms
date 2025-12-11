#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from geopandas import GeoDataFrame, read_file
from geopandas.testing import assert_geodataframe_equal
from shapely import LineString, Point

from geogenalg.application.generalize_fences import GeneralizeFences
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_fences_50k(testdata_path: Path) -> None:
    """Test generalizing fences with parameters to the 1: 50 000 scale."""
    input_path = testdata_path / "fences_rovaniemi.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="mtk_fences",
    )

    masts_data = read_file(input_path, layer="mtk_masts")

    temp_dir = TemporaryDirectory()
    output_path = temp_dir.name + "/generalized_fences.gpkg"

    algorithm = GeneralizeFences(
        closing_fence_area_threshold=2000,
        closing_fence_area_with_mast_threshold=8000,
        fence_length_threshold=80,
        fence_length_threshold_in_closed_area=300,
        simplification_tolerance=4,
        gap_threshold=25,
        attribute_for_line_merge="kohdeluokka",
    )

    control = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="generalized_fences",
    )
    algorithm.execute(input_data, reference_data={"masts": masts_data}).to_file(
        output_path, layer="result"
    )

    result = read_gdf_from_file_and_set_index(
        output_path, UNIQUE_ID_COLUMN, layer="result"
    )

    control = control.sort_values("geometry").reset_index(drop=True)
    result = result.sort_values("geometry").reset_index(drop=True)
    assert_geodataframe_equal(control, result)


def test_generalize_fences_50k_invalid_geometry_type() -> None:
    input_data = GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])

    algorithm = GeneralizeFences()

    with pytest.raises(
        GeometryTypeError,
        match=r"GeneralizeFences works only with LineString geometries.",
    ):
        algorithm.execute(data=input_data, reference_data={})


def test_generalize_fences_50k_missing_masts_data(testdata_path: Path) -> None:
    input_path = testdata_path / "fences_rovaniemi.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="mtk_fences",
    )

    algorithm = GeneralizeFences()

    with pytest.raises(
        KeyError,
        match=r"GeneralizeFences requires mast Point GeoDataFrame in reference_data with key 'masts'.",
    ):
        algorithm.execute(data=input_data, reference_data={})


def test_generalize_fences_50k_invalid_geometry_type_masts(testdata_path: Path) -> None:
    input_path = testdata_path / "fences_rovaniemi.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="mtk_fences",
    )
    masts_data = GeoDataFrame(
        {"id": [1]}, geometry=[LineString((Point(0, 0), Point(1, 0)))]
    )

    algorithm = GeneralizeFences()

    with pytest.raises(
        GeometryTypeError, match=r"Masts data should be a Point GeoDataFrame."
    ):
        algorithm.execute(data=input_data, reference_data={"masts": masts_data})
