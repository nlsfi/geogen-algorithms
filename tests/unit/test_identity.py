#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from shapely import Point

from geogenalg.core.exceptions import DuplicateIndexError
from geogenalg.identity import (
    hash_duplicate_indexes,
    hash_index_from_geometry,
    hash_index_from_old_ids,
)


@pytest.mark.parametrize(
    ("input_data", "hash_prefix", "control"),
    [
        pytest.param(
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
            id="no_duplicates",
        ),
        pytest.param(
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
                        "871f7ac5f093da508f60cffd723e31e9980a37d00909c851e0a879a6b77e07df",
                        "3b821fa1cee4c54be4c55eba06467208f3ae5e846bed3cd4f6023d82224902bc",
                    ]
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                ],
            ),
            id="duplicates",
        ),
        pytest.param(
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
                        "871f7ac5f093da508f60cffd723e31e9980a37d00909c851e0a879a6b77e07df",
                        "3b821fa1cee4c54be4c55eba06467208f3ae5e846bed3cd4f6023d82224902bc",
                        "2",
                    ]
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 0),
                    Point(2, 0),
                ],
            ),
            id="mixed",
        ),
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


def test_hash_duplicate_indexes_raises():
    data = GeoDataFrame(
        geometry=[
            Point(0, 0),
            Point(0, 0),
        ],
        index=["1", "1"],
    )

    with pytest.raises(DuplicateIndexError, match="Duplicate indexes found"):
        hash_duplicate_indexes(data, "test")


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


@pytest.mark.parametrize(
    ("input_data", "hash_prefix", "control"),
    [
        pytest.param(
            GeoDataFrame(geometry=[Point(0, 0)]),
            "test",
            GeoDataFrame(
                index=[
                    "83df0b76c9ba468466e2e3c790555534231cb9c2ceb9ab49e4d2ae28318873f3"
                ],
                geometry=[Point(0, 0)],
            ),
            id="point",
        ),
        pytest.param(
            GeoDataFrame(geometry=[]),
            "test",
            GeoDataFrame(
                geometry=[],
            ),
            id="empty",
        ),
        pytest.param(
            GeoDataFrame(
                geometry=[Point(5, 0)],
            ),
            "other",
            GeoDataFrame(
                index=[
                    "60f63b0c98add5b885666e90e5a8a04d7623a8412ebc605b58581cd041561865"
                ],
                geometry=[Point(5, 0)],
            ),
            id="hash_prefix",
        ),
    ],
)
def test_hash_index_from_geometry(
    input_data: GeoDataFrame,
    hash_prefix: str,
    control: GeoDataFrame,
):
    assert_geodataframe_equal(
        hash_index_from_geometry(input_data, hash_prefix),
        control,
    )
