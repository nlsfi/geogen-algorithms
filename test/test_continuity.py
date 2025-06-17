#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections import defaultdict

import pytest
from shapely.geometry import LineString, MultiLineString

from geogenalg import continuity


@pytest.mark.parametrize(
    ("lines", "expected_results"),
    [
        (
            [LineString([(0, 0), (1, 1)])],
            [((0, 0), 1), ((1, 1), 1)],
        ),
        (
            [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
            [((0, 0), 1), ((1, 1), 2), ((2, 2), 1)],
        ),
        (
            [MultiLineString([[(0, 0), (0.5, 0.5), (1, 0)], [(1, 0), (2, 0)]])],
            [((0, 0), 1), ((1, 0), 2), ((2, 0), 1)],
        ),
        (
            [LineString([])],
            [],
        ),
    ],
    ids=["single_line", "shared_endpoint", "multi_line", "empty_line"],
)
def test_find_all_endpoints(lines: list, expected_results: list):
    result = continuity.find_all_endpoints(lines)
    coord_count = defaultdict(int)
    for pt, _, count in result:
        key = (round(pt.x, 5), round(pt.y, 5))
        coord_count[key] = count

    expected_dict = dict(expected_results)

    assert len(coord_count) == len(expected_dict)
    for coord, count in expected_dict.items():
        assert coord in coord_count
        assert coord_count[coord] == count
