#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pandas.testing import assert_frame_equal
from pygeoops import GeoDataFrame
from shapely import Point

from geogenalg.application.generalize_watercourse_areas import (
    GeneralizeWaterCourseAreas,
)
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_watercourse_areas(testdata_path: Path):
    input_path = testdata_path / "watercourse_areas.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="watercourse_part_area",
    )

    temp_dir = TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    algorithm = GeneralizeWaterCourseAreas()

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


def test_invalid_geom_type() -> None:
    gdf = GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])

    with pytest.raises(
        GeometryTypeError,
        match=re.escape(
            "Input data must contain only geometries of following types: Polygon."
        ),
    ):
        GeneralizeWaterCourseAreas().execute(data=gdf, reference_data={})
