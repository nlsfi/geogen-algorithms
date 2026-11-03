#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
import re

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

from geogenalg.split import (
    explode_and_hash_id,
    split_lines_by_points,
)


@pytest.mark.parametrize(
    ("input_data", "hash_prefix", "control"),
    [
        pytest.param(
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
            id="no_multigeoms_no_changes",
        ),
        pytest.param(
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
                        "871f7ac5f093da508f60cffd723e31e9980a37d00909c851e0a879a6b77e07df",
                        "9e89f7bda054932a535101e15fa8d6ab53892da1752ad2229b1a6f60d9e9abfd",
                    ],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 1),
                ],
            ),
            id="one_multipoint_two_parts",
        ),
        pytest.param(
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
                        "871f7ac5f093da508f60cffd723e31e9980a37d00909c851e0a879a6b77e07df",
                        "9e89f7bda054932a535101e15fa8d6ab53892da1752ad2229b1a6f60d9e9abfd",
                        "91202388913c3f60756ea0a404421c48d2c4b5b87e86166df35b7ecfccf2f4f4",
                    ],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 1),
                    Point(2, 2),
                ],
            ),
            id="one_multipoint_three_parts",
        ),
        pytest.param(
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
                        "871f7ac5f093da508f60cffd723e31e9980a37d00909c851e0a879a6b77e07df",
                        "9e89f7bda054932a535101e15fa8d6ab53892da1752ad2229b1a6f60d9e9abfd",
                        "91202388913c3f60756ea0a404421c48d2c4b5b87e86166df35b7ecfccf2f4f4",
                        "2",
                        "1d8bf353b75b14064144e900b50430fb0d22432f8280d5092935f0da08fdbac3",
                        "0b8d7a9b498409a21bc60be6bc192e1a1a7a648bf7a901e2e79429d5afa202df",
                        "e9ed104fd65b2059f9bd2708f7964b3c3f6878ae856610fef04147cbecebfa9f",
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
            id="mixed_single_and_multis",
        ),
        pytest.param(
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
                        "c1a4c5a243b7a642d3bb522345e90171717fcf434cdcc9e0b0c2f3a407a1745a",
                        "dea7666371eca0045e8eb8a78adfdc017680de48bc75bda0c626fe4d67f29195",
                    ],
                },
                geometry=[
                    Point(0, 0),
                    Point(1, 1),
                ],
            ),
            id="one_multipoint_two_parts_different_prefix",
        ),
        pytest.param(
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
                        "3f4341180a26216acc624596d91343cc1fd966fd9649e86f7a74bc67670fb9b4",
                        "c916b6bd18f955ea8f87900ce986226fc89a04f06b54d079d9b492df3f24442d",
                    ],
                },
                geometry=[
                    LineString([[0, 0], [1, 1]]),
                    LineString([[5, 5], [6, 6]]),
                ],
            ),
            id="multiline",
        ),
        pytest.param(
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
                        "fb6570e7655a1034bb54fc7b075e9a1e5fc0809c85dcbed7f1702fdfcc4a406f",
                        "37684460fa28c9b7a6cb5defe44ebb753f73cf3c2b05a3023747e240f6500745",
                    ],
                },
                geometry=[
                    box(0, 0, 1, 1),
                    box(5, 5, 6, 6),
                ],
            ),
            id="multipolygon",
        ),
        pytest.param(
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
            id="multipoint_with_one_part_turns_to_point",
        ),
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


def test_explode_and_hash_id_raises():
    with pytest.raises(
        ValueError, match=re.escape("GeoDataFrame must have a string index.")
    ):
        explode_and_hash_id(GeoDataFrame(index=[1]), "")


@pytest.mark.parametrize(
    (
        "lines_gdf",
        "points_gdf",
        "max_distance",
        "expected_gdf",
    ),
    [
        (
            GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            [0, 0],
                            [10, 0],
                        ]
                    )
                ]
            ),
            GeoDataFrame(
                geometry=[
                    Point(5, 0),
                ]
            ),
            0.1,
            GeoDataFrame(
                geometry=[
                    MultiLineString(
                        [
                            LineString([[0, 0], [5, 0]]),
                            LineString([[5, 0], [10, 0]]),
                        ]
                    )
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            [0, 0],
                            [5, 0],
                            [10, 0],
                        ]
                    )
                ]
            ),
            GeoDataFrame(
                geometry=[
                    Point(5, 0),
                ]
            ),
            0.1,
            GeoDataFrame(
                geometry=[
                    MultiLineString(
                        [
                            LineString([[0, 0], [5, 0]]),
                            LineString([[5, 0], [10, 0]]),
                        ]
                    )
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            [0, 0],
                            [10, 0],
                        ]
                    )
                ]
            ),
            GeoDataFrame(
                geometry=[
                    Point(2, 0),
                    Point(7, 0),
                ]
            ),
            0.1,
            GeoDataFrame(
                geometry=[
                    MultiLineString(
                        [
                            LineString([[0, 0], [2, 0]]),
                            LineString([[2, 0], [7, 0]]),
                            LineString([[7, 0], [10, 0]]),
                        ]
                    )
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            [0, 0],
                            [10, 0],
                        ]
                    )
                ]
            ),
            GeoDataFrame(
                geometry=[
                    Point(5, 0.5),
                ]
            ),
            1.0,
            GeoDataFrame(
                geometry=[
                    MultiLineString(
                        [
                            LineString([[0, 0], [5, 0]]),
                            LineString([[5, 0], [10, 0]]),
                        ]
                    )
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            [0, 0],
                            [10, 0],
                        ]
                    )
                ]
            ),
            GeoDataFrame(
                geometry=[
                    Point(5, 2),
                ]
            ),
            1.0,
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [10, 0]]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            [0, 0],
                            [10, 0],
                        ]
                    )
                ]
            ),
            GeoDataFrame(
                geometry=[
                    Point(0, 0),
                    Point(10, 0),
                ]
            ),
            0.1,
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [10, 0]]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            [0, 0],
                            [5, 0],
                            [10, 0],
                        ]
                    )
                ]
            ),
            GeoDataFrame(
                geometry=[
                    Point(5, 0),
                    Point(5, 0),
                ]
            ),
            0.1,
            GeoDataFrame(
                geometry=[
                    MultiLineString(
                        [
                            LineString([[0, 0], [5, 0]]),
                            LineString([[5, 0], [10, 0]]),
                        ]
                    )
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString([[0, 0], [10, 0]]),
                    LineString([[0, 10], [10, 10]]),
                ],
            ),
            GeoDataFrame(
                geometry=[
                    Point(5, 0),
                    Point(2, 10),
                ]
            ),
            0.1,
            GeoDataFrame(
                geometry=[
                    MultiLineString(
                        [
                            LineString([[0, 0], [5, 0]]),
                            LineString([[5, 0], [10, 0]]),
                        ]
                    ),
                    MultiLineString(
                        [
                            LineString([[0, 10], [2, 10]]),
                            LineString([[2, 10], [10, 10]]),
                        ]
                    ),
                ],
            ),
        ),
    ],
    ids=[
        "single split",
        "split at existing vertex",
        "multiple splits",
        "snap nearby point",
        "point too far away",
        "ignore endpoints",
        "duplicate split points",
        "multiple lines",
    ],
)
def test_split_lines_by_points(
    lines_gdf: GeoDataFrame,
    points_gdf: GeoDataFrame,
    max_distance: float,
    expected_gdf: GeoDataFrame,
):
    result = split_lines_by_points(
        lines_gdf,
        points_gdf,
        max_distance=max_distance,
    )

    assert_geodataframe_equal(result, expected_gdf)
