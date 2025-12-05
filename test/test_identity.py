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

from geogenalg.identity import hash_duplicate_indexes, hash_index_from_old_ids


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


@pytest.mark.parametrize(
    ("input_data", "hash_prefix", "drop_old_ids", "control"),
    [
        (
            GeoDataFrame(
                {
                    "id": ["1", "2", "3"],
                    "old_ids": [("9", "10"), None, ("3",)],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                    Point(2, 0),
                ],
            ),
            "test",
            True,
            GeoDataFrame(
                {
                    "id": [
                        "6d54142c2eee9dd273f8b512a4cb7324a361b6e8444945c51458b14d407dce0e",
                        "2",
                        "3",
                    ],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                    Point(2, 0),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": ["1", "2", "3"],
                    "old_ids": [("9", "10"), None, ("3",)],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                    Point(2, 0),
                ],
            ),
            "test",
            False,
            GeoDataFrame(
                {
                    "id": [
                        "6d54142c2eee9dd273f8b512a4cb7324a361b6e8444945c51458b14d407dce0e",
                        "2",
                        "3",
                    ],
                    "old_ids": [("9", "10"), None, ("3",)],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                    Point(2, 0),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": ["1", "2", "3"],
                    "old_ids": [("9", "10"), None, ("3",)],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                    Point(2, 0),
                ],
            ),
            "different",
            True,
            GeoDataFrame(
                {
                    "id": [
                        "a432ccc62ec30628f7953fe5123fa1a8765d8bb28e3d9a38bbeb80309c6efd90",
                        "2",
                        "3",
                    ],
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
        "one hash value",
        "one hash value, different hash prefix",
        "one hash value, don't drop old ids",
    ],
)
def test_hash_index_from_old_ids(
    input_data: GeoDataFrame,
    hash_prefix: str,
    drop_old_ids: bool,
    control: GeoDataFrame,
):
    input_data = input_data.set_index("id")
    control = control.set_index("id")

    result = hash_index_from_old_ids(
        input_data,
        hash_prefix,
        "old_ids",
        drop_old_ids=drop_old_ids,
    )

    assert_geodataframe_equal(
        result,
        control,
    )
