#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from collections.abc import Callable

import pytest
from geopandas import GeoDataFrame
from pytest_benchmark.fixture import BenchmarkFixture

from geogenalg.exaggeration import exaggerate_thin_polygons


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "rectangle_count",
    [
        1000,
        10000,
        50000,
    ],
)
def test_exaggerate_thin_polygons(
    rectangle_count: int,
    benchmark: BenchmarkFixture,
    make_thin_and_wide_rects: Callable[[int], GeoDataFrame],
):
    rectangles = make_thin_and_wide_rects(rectangle_count)

    result = benchmark(
        exaggerate_thin_polygons,
        rectangles,
        width_threshold=1,
        elongation_threshold=0.25,
        exaggerate_by=1,
    )

    exaggerated = result.geometry.area > rectangles.geometry.area
    assert exaggerated.sum() == rectangle_count // 2 + rectangle_count % 2
