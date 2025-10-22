#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path

from geopandas import read_file
from pandas.testing import assert_frame_equal

from geogenalg.application.generalize_water_areas import GeneralizeWaterAreas
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_lakes(testdata_path: Path):
    input_path = testdata_path / "lakes.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="lake_part",
    )

    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    algorithm = GeneralizeWaterAreas(
        min_area=4000.0,
        min_island_area=100.0,
        max_simplify_tolerance=10.0,
        thin_section_width=20.0,
        thin_section_min_size=200.0,
        thin_section_exaggerate_by=3.0,
        island_min_width=185.0,
        island_min_elongation=0.25,
        island_exaggerate_by=3.0,
        smoothing_passes=3,
    )

    control = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="control",
    )
    algorithm.execute(input_data, {}).to_file(output_path, layer="result")

    result = read_gdf_from_file_and_set_index(
        output_path, UNIQUE_ID_COLUMN, layer="result"
    )

    assert_frame_equal(control, result)


def test_generalize_sea(testdata_path: Path):
    input_path = testdata_path / "sea.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="sea_part",
    )

    shoreline = read_file(input_path, layer="shoreline")

    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    algorithm = GeneralizeWaterAreas(
        min_area=4000.0,
        min_island_area=100.0,
        max_simplify_tolerance=10.0,
        thin_section_width=20.0,
        thin_section_min_size=200.0,
        thin_section_exaggerate_by=3.0,
        island_min_width=185.0,
        island_min_elongation=0.25,
        island_exaggerate_by=3.0,
        smoothing_passes=3,
    )

    control = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="control",
    )
    algorithm.execute(input_data, {"shoreline": shoreline}).to_file(
        output_path, layer="result"
    )

    result = read_gdf_from_file_and_set_index(
        output_path, UNIQUE_ID_COLUMN, layer="result"
    )

    assert_frame_equal(control, result)
