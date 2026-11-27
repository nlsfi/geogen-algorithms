from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import pytest
from geopandas import GeoDataFrame, GeoSeries
from geopandas.testing import assert_geoseries_equal
from pandas import Series
from pandas.testing import assert_series_equal
from shapely import (
    STRtree,
    area,
    box,
    buffer,
    equals,
    intersection,
    intersects,
    simplify,
)

from geogenalg.core.geometry import centerline
from geogenalg.core.parallel import in_parallel, num_workers, query_tree

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture

    from geogenalg.testing import TestInputData


@pytest.mark.benchmark
@pytest.mark.parametrize("feature_count", [10, 100, 1_000, 10_000])
@pytest.mark.parametrize(
    "mode", ["no_pool", "threadmanager_pool", "non_cached_threadpool"]
)
def test_benchmark_buffer(
    benchmark: BenchmarkFixture,
    make_point_gdf: Callable[..., GeoDataFrame],
    feature_count: int,
    mode: str,
) -> None:
    geoms = make_point_gdf(feature_count).geometry

    if mode == "no_pool":

        def run_process() -> GeoSeries:
            return GeoSeries(buffer(geoms.to_numpy(), 5.0, quad_segs=16), crs=geoms.crs)

    elif mode == "threadmanager_pool":

        def run_process() -> Series | GeoSeries:
            return in_parallel(geoms).buffer(5.0).resolve_geoseries()

    else:  # non_cached_threadpool

        def run_process() -> Series | GeoSeries:
            with ThreadPoolExecutor(max_workers=num_workers()) as fresh_pool:
                return (
                    in_parallel(geoms, executor=fresh_pool)
                    .buffer(5.0)
                    .resolve_geoseries()
                )

    result = benchmark(run_process)
    assert_geoseries_equal(result, geoms.buffer(5.0))


@pytest.mark.benchmark
@pytest.mark.parametrize("feature_count", [10, 100, 1_000, 10_000])
@pytest.mark.parametrize(
    "mode", ["no_pool", "threadmanager_pool", "non_cached_threadpool"]
)
def test_benchmark_intersects_single_geom(
    benchmark: BenchmarkFixture,
    make_overlapping_polygons: Callable[..., GeoDataFrame],
    feature_count: int,
    mode: str,
) -> None:
    geoms = make_overlapping_polygons(feature_count).geometry
    target_box = box(0.0, 0.0, 50.0, 50.0)

    if mode == "no_pool":

        def run_process() -> Series:
            return Series(intersects(geoms.to_numpy(), target_box))

    elif mode == "threadmanager_pool":

        def run_process() -> Series | GeoSeries:
            return in_parallel(geoms)._run(intersects, target_box).resolve_series()

    else:  # non_cached_threadpool

        def run_process() -> Series | GeoSeries:
            with ThreadPoolExecutor(max_workers=num_workers()) as fresh_pool:
                return (
                    in_parallel(geoms, executor=fresh_pool)
                    ._run(intersects, target_box)
                    .resolve_series()
                )

    result = benchmark(run_process)
    assert_series_equal(result, geoms.intersects(target_box))


@pytest.mark.benchmark
@pytest.mark.parametrize("feature_count", [10, 100, 1_000, 10_000])
@pytest.mark.parametrize(
    "mode", ["no_pool", "threadmanager_pool", "non_cached_threadpool"]
)
def test_benchmark_intersection_pairwise(
    benchmark: BenchmarkFixture,
    make_overlapping_polygons: Callable[..., GeoDataFrame],
    feature_count: int,
    mode: str,
) -> None:
    geoms_1 = make_overlapping_polygons(feature_count, seed=0).geometry
    geoms_2 = geoms_1.translate(xoff=0.5, yoff=0.5)

    if mode == "no_pool":

        def run_process() -> GeoSeries:
            return GeoSeries(intersection(geoms_1, geoms_2), crs=geoms_1.crs)

    elif mode == "threadmanager_pool":

        def run_process() -> Series | GeoSeries:
            return in_parallel(geoms_1).intersection(geoms_2).resolve_geoseries()

    else:  # non_cached_threadpool

        def run_process() -> Series | GeoSeries:
            with ThreadPoolExecutor(max_workers=num_workers()) as fresh_pool:
                return (
                    in_parallel(geoms_1, executor=fresh_pool)
                    .intersection(geoms_2)
                    .resolve_geoseries()
                )

    result = benchmark(run_process)
    assert_geoseries_equal(result, geoms_1.intersection(geoms_2))


@pytest.mark.benchmark
@pytest.mark.parametrize("feature_count", [10, 100, 1_000, 10_000])
@pytest.mark.parametrize(
    "mode", ["no_pool", "threadmanager_pool", "non_cached_threadpool"]
)
def test_benchmark_area(
    benchmark: BenchmarkFixture,
    make_overlapping_polygons: Callable[..., GeoDataFrame],
    feature_count: int,
    mode: str,
) -> None:
    geoms = make_overlapping_polygons(feature_count).geometry

    if mode == "no_pool":

        def run_process() -> np.ndarray:
            return area(geoms.to_numpy())

    elif mode == "threadmanager_pool":

        def run_process() -> Series | GeoSeries:
            return in_parallel(geoms)._run(area).resolve_geoseries()

    else:  # non_cached_threadpool

        def run_process() -> Series | GeoSeries:
            with ThreadPoolExecutor(max_workers=num_workers()) as fresh_pool:
                return (
                    in_parallel(geoms, executor=fresh_pool)
                    ._run(area)
                    .resolve_geoseries()
                )

    result = benchmark(run_process)
    assert len(result) == feature_count


@pytest.mark.benchmark
@pytest.mark.parametrize("feature_count", [10, 100, 1000, 10_000])
@pytest.mark.parametrize(
    "mode",
    ["no_pool", "threadmanager_pool", "non_cached_threadpool"],
)
def test_process_with_multiple_steps(
    benchmark: BenchmarkFixture,
    make_point_gdf: Callable[..., GeoDataFrame],
    feature_count: int,
    mode: str,
):
    geoms = make_point_gdf(feature_count).geometry

    if mode == "no_pool":

        def run_process() -> np.ndarray:
            step1 = buffer(geoms.to_numpy(), 5.0, quad_segs=16)
            step2 = simplify(step1, tolerance=0.5)
            return buffer(step2, 5.0)

    elif mode == "threadmanager_pool":

        def run_process() -> Series | GeoSeries:
            return (
                in_parallel(geoms)
                .buffer(5.0)
                ._run(simplify, tolerance=0.5)
                .buffer(5.0)
                .resolve_geoseries()
            )

    else:  # non_cached_threadpool

        def run_process() -> Series | GeoSeries:
            with ThreadPoolExecutor(max_workers=num_workers()) as fresh_pool:
                return (
                    in_parallel(geoms, executor=fresh_pool)
                    .buffer(5.0)
                    ._run(simplify, tolerance=0.5)
                    .buffer(5.0)
                    .resolve_geoseries()
                )

    result = benchmark(run_process)
    assert len(result) == feature_count


@pytest.mark.benchmark
@pytest.mark.parametrize("segmentize_by", [2, 10])
@pytest.mark.parametrize(
    "mode", ["no_pool", "threadmanager_pool", "non_cached_threadpool"]
)
def test_benchmark_centerline_watercourse_areas(
    benchmark: BenchmarkFixture,
    watercourse_areas_input: TestInputData,
    segmentize_by: float,
    mode: str,
) -> None:
    """Real watercourse area polygons."""
    polygons, _, _ = watercourse_areas_input.read()
    polygons = polygons.loc[polygons.geometry.geom_type == "Polygon"].geometry

    if mode == "no_pool":

        def run_process() -> Series | GeoSeries:
            return GeoSeries(
                [
                    centerline(polygon, segmentize_by=segmentize_by)
                    for polygon in polygons
                ],
                index=polygons.index,
                crs=polygons.crs,
            )

    elif mode == "threadmanager_pool":

        def run_process() -> Series | GeoSeries:
            return (
                in_parallel(polygons)
                .centerline(segmentize_by=segmentize_by)
                .resolve_geoseries()
            )

    else:  # non_cached_threadpool

        def run_process() -> Series | GeoSeries:
            with ThreadPoolExecutor(max_workers=num_workers()) as fresh_pool:
                return (
                    in_parallel(polygons, executor=fresh_pool)
                    .centerline(segmentize_by=segmentize_by)
                    .resolve_geoseries()
                )

    result = benchmark(run_process)
    expected = [
        centerline(polygon, segmentize_by=segmentize_by) for polygon in polygons
    ]
    assert equals(result.to_numpy(), expected).all()


@pytest.mark.benchmark
@pytest.mark.parametrize("feature_count", [10, 100, 1_000, 10_000, 100_000])
@pytest.mark.parametrize(
    "mode", ["no_pool", "threadmanager_pool", "non_cached_threadpool"]
)
def test_benchmark_query_tree(
    benchmark: BenchmarkFixture,
    make_overlapping_polygons: Callable[..., GeoDataFrame],
    feature_count: int,
    mode: str,
) -> None:
    geoms = make_overlapping_polygons(feature_count).geometry.to_numpy()
    tree = STRtree(geoms)

    if mode == "no_pool":

        def run_process() -> np.ndarray:
            return tree.query(geoms, predicate="intersects")

    elif mode == "threadmanager_pool":

        def run_process() -> np.ndarray:
            return query_tree(tree, geoms, predicate="intersects")

    else:  # non_cached_threadpool

        def run_process() -> np.ndarray:
            with ThreadPoolExecutor(max_workers=num_workers()) as fresh_pool:
                return query_tree(
                    tree,
                    geoms,
                    predicate="intersects",
                    executor=fresh_pool,
                )

    result = benchmark(run_process)
    np.testing.assert_array_equal(result, tree.query(geoms, predicate="intersects"))
