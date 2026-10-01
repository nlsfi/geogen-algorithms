#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from collections.abc import Callable

import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from pytest_benchmark.fixture import BenchmarkFixture
from shapely import get_num_coordinates
from shapely.geometry import LineString, Point

from geogenalg.core.geometry import (
    assign_z_from_attribute,
    concatenate_lines,
    gaussian_smooth,
    get_line_connection_points,
    get_topological_points,
)
from geogenalg.testing import GeoPackagePath, TestInputData


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("layer", "expected_points"),
    [
        ("100", 46),
        ("1000", 469),
        ("5000", 2036),
        ("10000", 3881),
    ],
    ids=[
        "100",
        "1000",
        "5000",
        "10000",
    ],
)
def test_get_topological_points(
    benchmark: BenchmarkFixture,
    line_network_input: GeoPackagePath,
    layer: str,
    expected_points: int,
):
    test_input = TestInputData(input_uri=line_network_input.to_input(layer))

    gdf, _, _ = test_input.read()

    result = benchmark(get_topological_points, gdf)

    assert len(result) == expected_points


def test_assign_z_from_attribute(benchmark: BenchmarkFixture):
    points = [Point(i, i) for i in range(10000)]
    z = list(range(10000))

    gdf = GeoDataFrame({"z": z}, geometry=points)

    result = benchmark(assign_z_from_attribute, gdf, "z")

    assert result.geometry.has_z.all()


def test_concatenate_lines(benchmark: BenchmarkFixture):
    n = 100000
    a = LineString([(i, i) for i in range(n)])
    b = LineString([(i, i) for i in range(n - 1, 2 * n - 1)])

    result = benchmark(concatenate_lines, a, b)

    assert get_num_coordinates(result) == 199999


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("layer", "expected_points"),
    [
        ("100", 46),
        ("1000", 469),
        ("5000", 2036),
        ("10000", 3881),
    ],
    ids=[
        "100",
        "1000",
        "5000",
        "10000",
    ],
)
def test_get_line_connection_points(
    benchmark: BenchmarkFixture,
    line_network_input: GeoPackagePath,
    layer: str,
    expected_points: int,
):
    test_input = TestInputData(input_uri=line_network_input.to_input(layer))

    gdf, _, _ = test_input.read()

    result = benchmark(get_line_connection_points, gdf)

    assert len(result) == expected_points


def test_get_topological_points_roads(
    benchmark: BenchmarkFixture,
    roads_input: TestInputData,
):
    # Test with road data to get a comparison against
    # get_line_connection_points(), as the synthetic benchmark data
    # consists of straight lines anyway largely eliminating the performance
    # benefit of it with datasets which have linestrings with more vertices.
    gdf, _, _ = roads_input.read()

    result = benchmark(get_topological_points, gdf)

    assert len(result) == 1050


def test_get_line_connection_points_roads(
    benchmark: BenchmarkFixture,
    roads_input: TestInputData,
):
    # Test with road data to get a comparison against get_topological_points(),
    # as the synthetic benchmark data consists of straight lines anyway largely
    # eliminating the performance benefit of this with datasets which have
    # linestrings with more vertices.
    gdf, _, _ = roads_input.read()

    result = benchmark(get_line_connection_points, gdf)

    assert len(result) == 1050


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "vertex_count",
    [
        100,
        1000,
        10000,
        30000,
        100000,
        1000000,
        10000000,
    ],
)
def test_gaussian_smooth(
    vertex_count: int,
    benchmark: BenchmarkFixture,
    make_linestring: Callable[[int], LineString],
):
    line = make_linestring(vertex_count)

    result = benchmark(gaussian_smooth, np.array([line]), sigma=2)

    assert len(result) == 1
    assert len(result[0].coords) == vertex_count
