#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from collections.abc import Callable

import pytest
from geopandas.geodataframe import GeoDataFrame
from pytest_benchmark.fixture import BenchmarkFixture

from geogenalg.selection import reduce_nearby_points_by_selecting


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("point_count", "after_reduction"),
    [
        (100, 39),
        (1000, 343),
        (5000, 1711),
        (10000, 3429),
    ],
    ids=[
        "100",
        "1000",
        "5000",
        "10000",
    ],
)
def test_reduce_nearby_points_by_selecting(
    benchmark: BenchmarkFixture,
    make_point_gdf: Callable[..., GeoDataFrame],
    point_count: int,
    after_reduction: int,
):
    gdf = make_point_gdf(point_count, seed=1)
    gdf["_id"] = gdf.index
    reference_gdf = make_point_gdf(point_count, seed=2)
    reference_gdf["_id"] = reference_gdf.index

    result = benchmark(
        reduce_nearby_points_by_selecting,
        gdf,
        reference_points_gdf=reference_gdf,
        distance_threshold=1.5,
    )

    assert result.shape[0] == after_reduction
