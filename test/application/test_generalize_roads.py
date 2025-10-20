#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path

import pytest
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
from shapely import Point, Polygon

from geogenalg.application.generalize_roads import (
    GeneralizeRoads,
)
from geogenalg.core.exceptions import GeometryTypeError

"""
@pytest.mark.parametrize(
    (
        "gpkg_file",
        "input_layer",
    ),
    [
        ("road_network.gpkg", "road_links"),
        ("path_network.gpkg", "paths"),
    ],
)"""


def test_generalize_roads(
    testdata_path: Path,
) -> None:
    input_path = testdata_path / "road_network.gpkg"
    input_data = read_file(input_path, layer="road_links")

    reference_input_path = testdata_path / "path_network.gpkg"

    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    algorithm = GeneralizeRoads(
        threshold_distance=10.0,
        threshold_length=75.0,
    )

    control = read_file(input_path, layer="control")
    reference_network = read_file(reference_input_path, layer="paths")
    algorithm.execute(input_data, {"paths": reference_network}).to_file(
        output_path, layer="result"
    )

    result = read_file(output_path, layer="result")

    assert_frame_equal(control, result)


def test_invalid_geom_type() -> None:
    gdf = GeoDataFrame(
        {"id": [1]},
        geometry=[
            Polygon((Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1), Point(0, 0)))
        ],
    )

    algorithm = GeneralizeRoads(
        threshold_distance=10.0,
        threshold_length=75.0,
    )

    with pytest.raises(
        GeometryTypeError,
        match="GeneralizeRoads works only with Linestring geometries",
    ):
        algorithm.execute(data=gdf, reference_data={})
