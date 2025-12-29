#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal

from geogenalg.application.generalize_shared_paths import GeneralizeSharedPaths
from geogenalg.testing import (
    GeoPackageInput,
    get_result_and_control,
)

UNIQUE_ID_COLUMN = "kmtk_id"


@pytest.mark.parametrize(
    (
        "gpkg_file",
        "input_layer",
    ),
    [
        ("roads.gpkg", "shared_path_link"),
        ("roads.gpkg", "path"),
    ],
    ids=[
        "shared",
        "path",
    ],
)
def test_generalize_shared_paths(
    testdata_path: Path,
    gpkg_file: str,
    input_layer: str,
) -> None:
    input_path = testdata_path / gpkg_file
    result, control = get_result_and_control(
        GeoPackageInput(input_path, layer_name=input_layer),
        GeoPackageInput(input_path, layer_name="control"),
        GeneralizeSharedPaths(
            detection_distance=25.0,
        ),
        UNIQUE_ID_COLUMN,
    )

    assert_frame_equal(result, control)
