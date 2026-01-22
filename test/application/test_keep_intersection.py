#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
from geopandas import GeoDataFrame
from pandas.testing import assert_frame_equal
from shapely.geometry import Point, Polygon

from geogenalg.application.keep_intersection import KeepIntersection
from geogenalg.core.exceptions import (
    GeometryTypeError,
    MissingReferenceError,
)
from geogenalg.testing import GeoPackageInput, get_result_and_control

UNIQUE_ID_COLUMN = "uuid"


@pytest.mark.parametrize(
    (
        "gpkg_file",
        "input_layer",
        "layer_suffix",
    ),
    [
        ("keep_intersection.gpkg", "data", "polygons"),
        ("keep_intersection.gpkg", "data", "lines"),
        ("keep_intersection.gpkg", "data", "points"),
    ],
    ids=[
        "polygons",
        "lines",
        "points",
    ],
)
def test_keep_intersection(
    testdata_path: Path,
    gpkg_file: str,
    input_layer: str,
    layer_suffix: str,
) -> None:
    input_layer = f"{input_layer}_{layer_suffix}"
    input_path = testdata_path / gpkg_file

    result, control = get_result_and_control(
        GeoPackageInput(input_path, layer_name=input_layer),
        GeoPackageInput(input_path, layer_name=f"control_{layer_suffix}"),
        KeepIntersection(),
        UNIQUE_ID_COLUMN,
        {
            "mask": GeoPackageInput(input_path, layer_name="mask"),
        },
    )

    assert_frame_equal(result, control)


def test_remove_overlap_missing_reference_data():
    with pytest.raises(
        MissingReferenceError,
        match=r"Reference data with mask polygons is mandatory.",
    ):
        KeepIntersection().execute(
            data=GeoDataFrame(
                {
                    "id": [1],
                },
                geometry=[Polygon([(0, 0), (0, 1), (1, 1)])],
                crs="EPSG:4326",
            ),
            reference_data={},
        )


def test_remove_overlap_invalid_mask_geometry():
    with pytest.raises(
        GeometryTypeError,
        match=r"Reference data must contain only geometries of following types: Polygon, MultiPolygon.",
    ):
        KeepIntersection().execute(
            data=GeoDataFrame(
                {
                    "id": [1],
                },
                geometry=[Polygon([(0, 0), (0, 1), (1, 1)])],
                crs="EPSG:4326",
            ),
            reference_data={
                "mask": GeoDataFrame(
                    {
                        "id": [1],
                    },
                    geometry=[Point(0, 0)],
                    crs="EPSG:4326",
                )
            },
        )
