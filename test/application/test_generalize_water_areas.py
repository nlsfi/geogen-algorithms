#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from geopandas import read_file
from pandas.testing import assert_frame_equal
from pygeoops import GeoDataFrame
from shapely import Point, box

from geogenalg.application.generalize_water_areas import GeneralizeWaterAreas
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_lakes(testdata_path: Path):
    input_path = testdata_path / "lakes.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="lake_part",
    )

    temp_dir = TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    algorithm = GeneralizeWaterAreas(
        min_area=4000.0,
        island_min_area=100.0,
        area_simplification_tolerance=10.0,
        thin_section_width=20.0,
        thin_section_min_size=200.0,
        thin_section_exaggerate_by=3.0,
        island_min_width=185.0,
        island_min_elongation=0.25,
        island_exaggerate_by=3.0,
        island_simplification_tolerance=10.0,
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

    assert_frame_equal(result, control)


def test_generalize_sea(testdata_path: Path):
    input_path = testdata_path / "sea.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="sea_part",
    )

    shoreline = read_file(input_path, layer="shoreline")

    temp_dir = TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    algorithm = GeneralizeWaterAreas(
        min_area=4000.0,
        area_simplification_tolerance=10.0,
        thin_section_width=20.0,
        thin_section_min_size=200.0,
        thin_section_exaggerate_by=3.0,
        island_min_area=100.0,
        island_min_width=185.0,
        island_min_elongation=0.25,
        island_exaggerate_by=3.0,
        island_simplification_tolerance=10.0,
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


def test_invalid_geom_type() -> None:
    gdf = GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])

    with pytest.raises(
        GeometryTypeError,
        match=re.escape(
            "Input data must contain only geometries of following types: Polygon."
        ),
    ):
        GeneralizeWaterAreas().execute(data=gdf, reference_data={})


def test_invalid_shoreline_geom_type() -> None:
    gdf = GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)])
    reference_gdf = GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])

    with pytest.raises(
        GeometryTypeError,
        match=re.escape(
            "Reference data must contain only geometries of following types: LineString."
        ),
    ):
        GeneralizeWaterAreas().execute(
            data=gdf, reference_data={"shoreline": reference_gdf}
        )
