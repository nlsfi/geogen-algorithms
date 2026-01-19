#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import re
from pathlib import Path

import pytest
from geopandas import GeoDataFrame
from pandas.testing import assert_frame_equal
from shapely import Point

from geogenalg.application.generalize_shared_paths import GeneralizeSharedPaths
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.testing import (
    GeoPackageInput,
    get_result_and_control,
)

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_shared_paths(
    testdata_path: Path,
) -> None:
    input_path = testdata_path / "roads.gpkg"

    result, control = get_result_and_control(
        GeoPackageInput(input_path, layer_name="shared_path_link"),
        GeoPackageInput(input_path, layer_name="control_shared_paths"),
        GeneralizeSharedPaths(
            detection_distance=25.0,
        ),
        UNIQUE_ID_COLUMN,
        # Use control from GeneralizeRoads test as reference, as these are intended
        # to be used sequentially.
        reference_uris={
            "roads": GeoPackageInput(input_path, layer_name="control"),
        },
    )

    assert_frame_equal(result, control)


def test_invalid_geom_type() -> None:
    with pytest.raises(
        GeometryTypeError,
        match=re.escape("Input data must contain only LineStrings."),
    ):
        GeneralizeSharedPaths().execute(
            GeoDataFrame(geometry=[Point(0, 0)]),
        )
