#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from shapely import Point

from geogenalg.identity import hash_duplicate_indexes


@pytest.mark.parametrize(
    ("input_data", "hash_prefix", "control"),
    [
        (
            GeoDataFrame(
                {
                    "id": ["1", "2"],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                ],
            ),
            "test",
            GeoDataFrame(
                {
                    "id": ["1", "2"],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": ["1", "1"],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                ],
            ),
            "test",
            GeoDataFrame(
                {
                    "id": [
                        "3d5dd0c0636dd82f73752f3040efc5d4ab6d87cf420b84e5f6cd33afaae910cb",
                        "b62cef672cfd20f774c4355ffd4c980868975d3fe2eafc6ddc52be551d745fc5",
                    ]
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": ["1", "1", "2"],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                    Point(2, 0),
                ],
            ),
            "test",
            GeoDataFrame(
                {
                    "id": [
                        "3d5dd0c0636dd82f73752f3040efc5d4ab6d87cf420b84e5f6cd33afaae910cb",
                        "b62cef672cfd20f774c4355ffd4c980868975d3fe2eafc6ddc52be551d745fc5",
                        "2",
                    ]
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                    Point(2, 0),
                ],
            ),
        ),
    ],
    ids=[
        "no duplicates",
        "duplicates",
        "mixed",
    ],
)
def test_hash_duplicate_indexes(
    input_data: GeoDataFrame,
    hash_prefix: str,
    control: GeoDataFrame,
):
    input_data = input_data.set_index("id")
    control = control.set_index("id")

    result = hash_duplicate_indexes(input_data, hash_prefix)

    assert_geodataframe_equal(
        result,
        control,
    )
