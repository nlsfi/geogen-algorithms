#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from collections.abc import Callable

import pytest
from geopandas.geodataframe import GeoDataFrame
from pytest_benchmark.fixture import BenchmarkFixture

from geogenalg.analyze import group_intersecting_geometries


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("polygon_count"),
    [
        (100),
        (1000),
        (5000),
        (10000),
    ],
    ids=[
        "100",
        "1000",
        "5000",
        "10000",
    ],
)
def test_group_intersecting_geometries(
    benchmark: BenchmarkFixture,
    make_overlapping_polygons: Callable[[int], GeoDataFrame],
    polygon_count: int,
):
    gdf = make_overlapping_polygons(polygon_count)

    result = benchmark(group_intersecting_geometries, gdf)

    assert result["_geometry_group"].value_counts().sum() == polygon_count
