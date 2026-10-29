#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from collections.abc import Callable
from math import isclose

import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from pytest_benchmark.fixture import BenchmarkFixture
from shapely import get_num_coordinates
from shapely.geometry import LineString, MultiPoint, Point

from geogenalg.core.geometry import (
    assign_nearest_z,
    assign_z_from_attribute,
    chaikin_smooth_keep_topology,
    concatenate_lines,
    extract_interior_rings_of_union,
    gaussian_smooth,
    get_line_connection_points,
    get_topological_points,
    substring,
    visvalingam_whyatt_simplify_conditional,
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


@pytest.mark.benchmark
def test_substring(
    benchmark: BenchmarkFixture,
    make_linestring: Callable[[int], LineString],
):
    line = make_linestring(1000)

    result = benchmark(substring, line, 40, 50)

    assert isclose(result.length, 10)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "geom_count",
    [
        100,
        1_000,
        5_000,
        20_000,
    ],
)
def test_assign_nearest_z(
    geom_count: int,
    benchmark: BenchmarkFixture,
    make_point_gdf: Callable[..., GeoDataFrame],
):
    source_gdf = make_point_gdf(geom_count, seed=0, has_z=True)
    target_gdf = source_gdf.copy()
    target_gdf.geometry = target_gdf.geometry.buffer(10, resolution=64)

    result = benchmark(assign_nearest_z, source_gdf, target_gdf)

    assert len(result) == geom_count
    assert result.geometry.has_z.all()


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "grid_size",
    [
        5,
        15,
        30,
    ],
)
def test_chaikin_smooth_keep_topology(
    grid_size: int,
    benchmark: BenchmarkFixture,
    make_polygon_grid: Callable[..., GeoDataFrame],
):
    polygons = make_polygon_grid(grid_size, vertices_per_edge=50)

    result = benchmark(
        chaikin_smooth_keep_topology,
        polygons.geometry,
        iterations=3,
        extra_skip_coords=None,
        distance_threshold=0.05,
    )

    assert len(result) == grid_size * grid_size


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "grid_size",
    [
        5,
        15,
        30,
    ],
)
@pytest.mark.parametrize(
    "use_skip_coords",
    [
        False,
        True,
    ],
    ids=[
        "no_skip_coords",
        "topological_points_as_skip_coords",
    ],
)
def test_visvalingam_whyatt_simplify_conditional(
    grid_size: int,
    use_skip_coords: bool,
    benchmark: BenchmarkFixture,
    make_polygon_grid: Callable[..., GeoDataFrame],
):
    polygons = make_polygon_grid(grid_size, vertices_per_edge=50)
    skip_coords = (
        MultiPoint(get_topological_points(polygons)) if use_skip_coords else None
    )

    result = benchmark(
        visvalingam_whyatt_simplify_conditional,
        polygons.geometry,
        1e-5,
        skip_coords=skip_coords,
    )

    assert len(result) == grid_size * grid_size


@pytest.mark.benchmark
@pytest.mark.parametrize(
    (
        "part_count",
        "gap_count",
        "vertices_per_gap",
        "expected_rings",
    ),
    [
        (5, 20, 200, 18),
        (20, 100, 200, 94),
        (50, 400, 200, 342),
    ],
)
def test_extract_interior_rings_of_union(
    part_count: int,
    gap_count: int,
    vertices_per_gap: int,
    expected_rings: int,
    benchmark: BenchmarkFixture,
    make_touching_polygons_with_holes: Callable[..., np.ndarray],
):
    geoms = make_touching_polygons_with_holes(part_count, gap_count, vertices_per_gap)

    result = benchmark(extract_interior_rings_of_union, geoms)

    assert len(result) == expected_rings
