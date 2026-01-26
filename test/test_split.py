#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from shapely import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    box,
)

from geogenalg.split import explode_and_hash_id


@pytest.mark.parametrize(
    ("input_data", "hash_prefix", "control"),
    [
        (
            GeoDataFrame(
                {
                    "id": ["1", "2", "3"],
                },
                geometry=[
                    Point(0, 0),
                    Point(0, 0),
                    Point(0, 0),
                ],
            ),
            "test",
            GeoDataFrame(
                {
                    "id": ["1", "2", "3"],
                },
                geometry=[
                    Point(0, 0),
                    Point(0, 0),
                    Point(0, 0),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [
                        "1",
                    ],
                },
                geometry=[
                    MultiPoint(
                        [
                            Point(0, 0),
                            Point(1, 1),
                        ]
                    ),
                ],
            ),
            "test",
            GeoDataFrame(
                {
                    "id": [
                        "3d5dd0c0636dd82f73752f3040efc5d4ab6d87cf420b84e5f6cd33afaae910cb",
                        "be6862c3e5ff8f99aea8204b5f1ff11b2908b3b95635f42b62468b03c0dcf3e2",
                    ],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 1),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [
                        "1",
                    ],
                },
                geometry=[
                    MultiPoint(
                        [
                            Point(0, 0),
                            Point(1, 1),
                            Point(2, 2),
                        ]
                    ),
                ],
            ),
            "test",
            GeoDataFrame(
                {
                    "id": [
                        "3d5dd0c0636dd82f73752f3040efc5d4ab6d87cf420b84e5f6cd33afaae910cb",
                        "be6862c3e5ff8f99aea8204b5f1ff11b2908b3b95635f42b62468b03c0dcf3e2",
                        "a2959cf4636524fd6b0389255d0d25b0760501dbe08239a479e1eff5fb3e0049",
                    ],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 1),
                    Point(2, 2),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [
                        "1",
                        "2",
                        "3",
                    ],
                },
                geometry=[
                    MultiPoint(
                        [
                            Point(0, 0),
                            Point(1, 1),
                            Point(2, 2),
                        ]
                    ),
                    Point(9, 9),
                    MultiPoint(
                        [
                            Point(3, 3),
                            Point(4, 4),
                            Point(5, 5),
                        ]
                    ),
                ],
            ),
            "test",
            GeoDataFrame(
                {
                    "id": [
                        "3d5dd0c0636dd82f73752f3040efc5d4ab6d87cf420b84e5f6cd33afaae910cb",
                        "be6862c3e5ff8f99aea8204b5f1ff11b2908b3b95635f42b62468b03c0dcf3e2",
                        "a2959cf4636524fd6b0389255d0d25b0760501dbe08239a479e1eff5fb3e0049",
                        "2",
                        "c27bcaf812f66788ce36b1f3c94931d63b8fabc7c405e4c57a0ae455e7c79661",
                        "13f166cb2ed7a4f5ec0de039b3ba7336281c9158ecbed796805614347b4d5453",
                        "634d6aa6cec90560aedc16300d457324ec42612d8be4b4455a8dab9c92b5f782",
                    ],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 1),
                    Point(2, 2),
                    Point(9, 9),
                    Point(3, 3),
                    Point(4, 4),
                    Point(5, 5),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [
                        "1",
                    ],
                },
                geometry=[
                    MultiPoint(
                        [
                            Point(0, 0),
                            Point(1, 1),
                        ]
                    ),
                ],
            ),
            "different",
            GeoDataFrame(
                {
                    "id": [
                        "9165fcfb53562882c625d9fc41347a09ea355829e3d2456913395c7aba1008f2",
                        "eb8956b3fd54ec54675df565c970539fa8bd61296dd129439eac8c83b46042b7",
                    ],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 1),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [
                        "1",
                    ],
                },
                geometry=[
                    MultiLineString(
                        [
                            LineString(
                                [
                                    (0, 0),
                                    (1, 1),
                                ]
                            ),
                            LineString(
                                [
                                    (5, 5),
                                    (6, 6),
                                ]
                            ),
                        ]
                    ),
                ],
            ),
            "test",
            GeoDataFrame(
                {
                    "id": [
                        "de14a495a2995cac1ce676ec94e834b6442a344c527c1461876fafdd84c521e7",
                        "51fd37f2d9ea7d1fb964fafd2fc79e7b096193513c0510fae159e5295f3ab747",
                    ],
                },
                geometry=[
                    LineString([[0, 0], [1, 1]]),
                    LineString([[5, 5], [6, 6]]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": [
                        "1",
                    ],
                },
                geometry=[
                    MultiPolygon(
                        [
                            box(0, 0, 1, 1),
                            box(5, 5, 6, 6),
                        ]
                    ),
                ],
            ),
            "test",
            GeoDataFrame(
                {
                    "id": [
                        "eb5fc6895e0390bb3e011d4f86baafbaf15999f0b835aef3e8a8cb7a9c3ccec1",
                        "7ca58054eae784dc147b1a986cf322d052c3299131bbc564d18742ebb17d57e0",
                    ],
                },
                geometry=[
                    box(0, 0, 1, 1),
                    box(5, 5, 6, 6),
                ],
            ),
        ),
        (
            GeoDataFrame(
                {
                    "id": ["1"],
                },
                geometry=[
                    MultiPoint([(0, 0)]),
                ],
            ),
            "test",
            GeoDataFrame(
                {
                    "id": ["1"],
                },
                geometry=[
                    Point(0, 0),
                ],
            ),
        ),
    ],
    ids=[
        "no multigeoms, no changes",
        "one multipoint, two parts",
        "one multipoint, three parts",
        "mixed single and multis",
        "one multipoint, two parts, different prefix",
        "multiline",
        "multipolygon",
        "multipoint with one part turns to point",
    ],
)
def test_explode_and_hash_id(
    input_data: GeoDataFrame,
    hash_prefix: str,
    control: GeoDataFrame,
):
    input_data = input_data.set_index("id")
    control = control.set_index("id")

    result = explode_and_hash_id(input_data, hash_prefix)

    assert_geodataframe_equal(
        result,
        control,
    )
