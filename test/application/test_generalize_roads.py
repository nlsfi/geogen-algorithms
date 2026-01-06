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
from shapely import Point, Polygon

from geogenalg.application.generalize_roads import GeneralizeRoads
from geogenalg.core.exceptions import GeometryTypeError
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
        ("roads.gpkg", "road_link"),
    ],
    ids=[
        "roads_to_generalize",
    ],
)
def test_generalize_roads(
    testdata_path: Path,
    gpkg_file: str,
    input_layer: str,
) -> None:
    input_path = testdata_path / gpkg_file
    result, control = get_result_and_control(
        GeoPackageInput(input_path, layer_name=input_layer),
        GeoPackageInput(input_path, layer_name="control"),
        GeneralizeRoads(
            threshold_distance=10.0,
            threshold_length=75.0,
            connection_info_column="is_connected",
        ),
        UNIQUE_ID_COLUMN,
        {"path": GeoPackageInput(input_path, layer_name="path")},
    )

    assert_frame_equal(result, control)


def test_invalid_geom_type() -> None:
    with pytest.raises(
        GeometryTypeError,
        match=re.escape("GeneralizeRoads works only with LineString geometries"),
    ):
        GeneralizeRoads().execute(
            GeoDataFrame(
                {"id": [1]},
                geometry=[
                    Polygon(
                        (
                            Point(0, 0),
                            Point(1, 0),
                            Point(1, 1),
                            Point(0, 1),
                            Point(0, 0),
                        )
                    )
                ],
            ),
        )
