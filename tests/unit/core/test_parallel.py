#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from geopandas import GeoSeries
from geopandas.testing import assert_geoseries_equal
from pandas import Index, Series
from shapely import (
    STRtree,
    box,
    buffer,
    convex_hull,
    equals,
    length,
)
from shapely.geometry import LineString, MultiPoint, Point

from geogenalg.core import parallel
from geogenalg.core.geometry import centerline
from geogenalg.core.parallel import (
    _THREAD_MANAGER,
    GEOGENALG_WORKER_COUNT_ENV_VAR,
    ParallelGeometryProcess,
    _get_chunk_slices,
    _process_arg,
    _process_geometry_array,
    get_threadpool_executor,
    in_parallel,
    num_workers,
    query_tree,
    set_num_workers,
    shutdown_threadpool_executor,
    union_all_by_group,
    union_all_by_row,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def reset_thread_manager() -> Generator[None, None, None]:
    shutdown_threadpool_executor()
    set_num_workers(None)
    yield
    shutdown_threadpool_executor()
    set_num_workers(None)


@pytest.fixture
def sample_points() -> GeoSeries:
    geoms = [Point(i, i) for i in range(12)]
    index = Index([f"idx_{i}" for i in range(12)], name="index")
    return GeoSeries(geoms, index=index, crs="EPSG:3857")


@pytest.mark.parametrize(
    ("value_count", "chunk_count", "expected"),
    [
        (10, 2, [slice(0, 5), slice(5, 10)]),
        (10, 3, [slice(0, 4), slice(4, 7), slice(7, 10)]),
        (5, 5, [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 4), slice(4, 5)]),
        (1, 1, [slice(0, 1)]),
        (0, 1, [slice(0, 0)]),
    ],
)
def test_get_chunk_slices(
    value_count: int,
    chunk_count: int,
    expected: list[slice],
):
    slices = _get_chunk_slices(value_count, chunk_count)
    assert slices == expected
    total_elements = sum(s.stop - s.start for s in slices)
    assert total_elements == value_count


@pytest.mark.parametrize(
    ("arg", "total_geom_count", "expected"),
    [
        pytest.param(
            np.array([10, 20, 30, 40]),
            4,
            [np.array([10]), np.array([20]), np.array([30]), np.array([40])],
            id="array_gets_sliced",
        ),
        pytest.param(
            np.array([10, 20, 30, 40]),
            10,
            [np.array([10, 20, 30, 40])] * 4,
            id="array_gets_broadcasted",
        ),
        pytest.param(
            Series([10, 20, 30, 40], index=["a", "b", "c", "d"]),
            4,
            [
                Series([10], index=["a"]),
                Series([20], index=["b"]),
                Series([30], index=["c"]),
                Series([40], index=["d"]),
            ],
            id="series_gets_sliced",
        ),
        pytest.param(
            Series([10, 20]),
            4,
            [Series([10, 20])] * 4,
            id="series_gets_broadcasted",
        ),
        pytest.param(
            5.0,
            4,
            [5.0, 5.0, 5.0, 5.0],
            id="scalar_float",
        ),
        pytest.param(
            "round",
            4,
            ["round", "round", "round", "round"],
            id="scalar_string",
        ),
        pytest.param(
            None,
            4,
            [None, None, None, None],
            id="scalar_none",
        ),
    ],
)
def test_process_arg(
    arg: object,
    total_geom_count: int,
    expected: list[object],
):
    for i in range(len(expected)):
        _slice = slice(i, i + 1)
        result = _process_arg(arg, _slice, total_geom_count)

        if isinstance(result, np.ndarray):
            np.testing.assert_array_equal(result, expected[i])
        elif isinstance(result, Series):
            assert result.equals(cast("Series", expected[i]))
        else:
            assert result == expected[i]


def test_worker_count_setter():
    set_num_workers(4)
    assert num_workers() == 4
    assert get_threadpool_executor()._max_workers == 4

    set_num_workers(2)
    assert num_workers() == 2
    assert get_threadpool_executor()._max_workers == 2


def test_worker_count_setter_raises():
    with pytest.raises(ValueError, match=re.escape("Value must be >= 1.")):
        set_num_workers(0)

    with pytest.raises(ValueError, match=re.escape("Value must be >= 1.")):
        set_num_workers(-5)


def test_worker_count_env_variable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(GEOGENALG_WORKER_COUNT_ENV_VAR, "3")
    set_num_workers(None)
    assert num_workers() == 3


@pytest.mark.parametrize(
    ("env_var", "warn_message"),
    [
        ("0", "must be > 0. Using default."),
        ("text", "could not be parsed as an integer. Using default."),
    ],
    ids=[
        "invalid_number",
        "not_a_number",
    ],
)
def test_worker_count_env_variable_warns(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    warn_message: str,
):
    monkeypatch.setenv(GEOGENALG_WORKER_COUNT_ENV_VAR, env_var)
    set_num_workers(None)
    with pytest.warns(UserWarning, match=re.escape(warn_message)):
        assert num_workers() >= 1


def test_executor_lifecycle_and_shutdown():
    executor = get_threadpool_executor()
    assert isinstance(executor, ThreadPoolExecutor)
    assert not executor._shutdown

    shutdown_threadpool_executor()
    assert _THREAD_MANAGER._executor is None

    new_executor = get_threadpool_executor()
    assert new_executor is not executor
    assert not new_executor._shutdown


def test_worker_setter_recreates_pool():
    executor1 = get_threadpool_executor()
    set_num_workers(3)
    executor2 = get_threadpool_executor()

    assert executor1 is not executor2
    assert executor1._shutdown
    assert executor2._max_workers == 3


@pytest.mark.parametrize(
    (
        "geoms",
        "workers",
        "use_custom_executor",
        "expect_parallel",
    ),
    [
        pytest.param(
            np.array([], dtype=object),
            4,
            False,
            False,
            id="empty_input",
        ),
        pytest.param(
            np.array([Point(0, 0)], dtype=object),
            4,
            False,
            False,
            id="single_input_runs_on_main",
        ),
        pytest.param(
            np.array([Point(0, 0), Point(1, 1)], dtype=object),
            1,
            False,
            False,
            id="set_single_worker_runs_on_main",
        ),
        pytest.param(
            np.array([Point(i, i) for i in range(4)], dtype=object),
            2,
            False,
            True,
            id="four_geoms_get_split_to_two_workers",
        ),
        pytest.param(
            np.array([Point(i, i) for i in range(7)], dtype=object),
            3,
            False,
            True,
            id="uneven_split_to_workers",
        ),
        pytest.param(
            np.array([Point(0, 0), Point(1, 1), Point(2, 2)], dtype=object),
            8,
            False,
            True,
            id="more_workers_than_geoms",
        ),
        pytest.param(
            np.array([Point(i, i) for i in range(10)], dtype=object),
            4,
            True,
            True,
            id="custom_executor",
        ),
    ],
)
def test_process_geometry_array(
    geoms: np.ndarray,
    workers: int,
    use_custom_executor: bool,
    expect_parallel: bool,
):
    set_num_workers(workers)

    main_thread_id = threading.get_ident()
    thread_ids: set[int] = set()
    lock = threading.Lock()

    def tracking_buffer(geoms: np.ndarray, distance: float) -> np.ndarray:
        with lock:
            thread_ids.add(threading.get_ident())
        return buffer(geoms, distance)

    pool_context = (
        ThreadPoolExecutor(max_workers=workers)
        if use_custom_executor
        else nullcontext()
    )

    with pool_context as custom_executor:
        result = _process_geometry_array(
            geoms,
            tracking_buffer,
            1,
            executor=custom_executor,
        )

    assert isinstance(result, np.ndarray)
    assert len(result) == len(geoms)
    assert result.dtype == object

    expected = buffer(geoms, 1)
    assert equals(result, expected).all()

    if len(geoms) == 0:
        assert len(thread_ids) == 0
    elif expect_parallel:
        assert len(thread_ids) >= 1
        assert main_thread_id not in thread_ids
    else:
        assert thread_ids == {main_thread_id}


def test_in_parallel(sample_points: GeoSeries):
    process = in_parallel(sample_points)

    assert isinstance(process, ParallelGeometryProcess)
    assert len(process._data) == len(sample_points)


def test_resolve_geoseries(sample_points: GeoSeries):
    process = in_parallel(sample_points).resolve_geoseries()

    assert isinstance(process, GeoSeries)
    assert_geoseries_equal(process, sample_points)


def test_resolve_numpy(sample_points: GeoSeries):
    result = in_parallel(sample_points)._run(length).resolve_numpy()

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(
        result,
        np.array([0.0 for _ in range(12)]),
    )


def test_resolve_series(sample_points: GeoSeries):
    process = in_parallel(sample_points)._run(length).resolve_series()

    assert isinstance(process, Series)
    assert (process == 0.0).all()


def test_buffer_with_scalar_distance(sample_points: GeoSeries):
    set_num_workers(4)
    result = in_parallel(sample_points).buffer(1.5).resolve_geoseries()

    assert isinstance(result, GeoSeries)
    assert result.crs == sample_points.crs
    assert result.index.equals(sample_points.index)

    expected = buffer(sample_points, 1.5, quad_segs=16)
    assert equals(result, expected).all()


def test_buffer_with_non_scalar_distance(sample_points: GeoSeries):
    set_num_workers(4)
    distances = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    result = in_parallel(sample_points).buffer(distances).resolve_geoseries()
    expected = buffer(sample_points, distances, quad_segs=16)

    assert equals(result, expected).all()


def test_intersection_against_single_geom(sample_points: GeoSeries):
    poly = box(-0.1, -0.1, 0.1, 0.1)

    result = in_parallel(sample_points).intersection(poly).resolve_geoseries()

    assert result.iloc[0] == Point(0, 0)

    for i in range(1, 12):
        assert result.iloc[i].is_empty


def test_difference_against_single_geom(sample_points: GeoSeries):
    poly = box(-0.1, -0.1, 0.1, 0.1)

    result = in_parallel(sample_points).difference(poly).resolve_geoseries()

    assert result.iloc[0].is_empty

    for i in range(1, 12):
        assert result.iloc[i] == Point(i, i)


def test_intersection_pairwise(sample_points: GeoSeries):
    boxes = np.array(
        [box(i - 0.1, i - 0.1, i + 0.1, i + 0.1) for i in range(len(sample_points))],
        dtype=object,
    )

    result = in_parallel(sample_points).intersection(boxes).resolve_geoseries()

    for i in range(len(sample_points)):
        assert result.iloc[i] == Point(i, i)


def test_difference_pairwise(sample_points: GeoSeries):
    boxes = np.array(
        [box(i - 0.1, i - 0.1, i + 0.1, i + 0.1) for i in range(len(sample_points))],
        dtype=object,
    )

    result = in_parallel(sample_points).difference(boxes).resolve_geoseries()

    for i in range(len(sample_points)):
        assert result.iloc[i].is_empty


def test_in_parallel_chaining(sample_points: GeoSeries):
    result = in_parallel(sample_points).buffer(1.0).buffer(5.0).resolve_geoseries()

    assert isinstance(result, GeoSeries)

    expected = sample_points.buffer(1.0).buffer(5.0)

    assert_geoseries_equal(result, expected)


def test_in_parallel_chaining_non_cached_pool(sample_points: GeoSeries):
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = (
            in_parallel(sample_points, executor=pool)
            .buffer(1.0)
            .buffer(5.0)
            .resolve_geoseries()
        )

        assert isinstance(result, GeoSeries)
        expected = sample_points.buffer(1.0).buffer(5.0)
        assert_geoseries_equal(result, expected)


def test_convex_hull():
    geoms = GeoSeries(
        [MultiPoint([(i, 0), (i + 1, 2), (i + 2, 0), (i + 1, 1)]) for i in range(10)],
        crs="EPSG:3067",
    )
    set_num_workers(4)

    result = in_parallel(geoms).convex_hull().resolve_geoseries()

    assert result.crs == geoms.crs
    assert result.index.equals(geoms.index)
    assert equals(result.to_numpy(), convex_hull(geoms.to_numpy())).all()


def test_within(sample_points: GeoSeries):
    boxes = np.array(
        [box(i - 1, i - 1, i + 1, i + 1) for i in range(len(sample_points))],
        dtype=object,
    )

    # All points are within a box, except this is moved away
    boxes[3] = box(100, 100, 101, 101)

    result = in_parallel(sample_points).within(boxes).resolve_numpy()

    assert result.dtype == bool

    # Check that boxes[3] is not within
    assert result.tolist() == [i != 3 for i in range(len(sample_points))]


def test_union_all_by_row():
    grid = np.array(
        [[box(0, 0, 1, 1), box(1, 0, 2, 1)] for _ in range(7)],
        dtype=object,
    )
    set_num_workers(3)

    result = union_all_by_row(grid)

    assert len(result) == 7
    assert all(equals(geom, box(0, 0, 2, 1)) for geom in result)


def test_in_parallel_with_numpy_array(sample_points: GeoSeries):
    geoms = sample_points.to_numpy()

    process = in_parallel(geoms)
    result = process.buffer(1.0).resolve_geoseries()

    assert result.crs is None
    assert result.index.equals(Index(range(len(geoms))))
    assert equals(result.to_numpy(), buffer(geoms, 1.0, quad_segs=16)).all()


def test_union_all_by_group():
    geoms = np.array([box(i, 0, i + 1, 1) for i in range(6)], dtype=object)
    groups = [np.array([0]), np.array([1, 2, 3]), np.array([4, 5]), np.array([2, 3])]
    set_num_workers(3)

    result = union_all_by_group(geoms, groups)

    assert len(result) == 4
    assert equals(
        result, [box(0, 0, 1, 1), box(1, 0, 4, 1), box(4, 0, 6, 1), box(2, 0, 4, 1)]
    ).all()


@pytest.mark.parametrize(
    "workers",
    [1, 3, 8],
    ids=["single_worker", "fewer_workers_than_geoms", "more_workers_than_geoms"],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="default_arguments"),
        pytest.param(
            {
                "simplify_tolerance": 0,
                "prune_branches_shorter_than": 2,
                "segmentize_by": 0.5,
                "merge_vertices_closer_than": 0.1,
            },
            id="all_arguments",
        ),
    ],
)
def test_centerline(workers: int, kwargs: dict):
    polygons = GeoSeries(
        [box(0, i * 10, 20 + i * 5, i * 10 + 2) for i in range(5)],
        index=Index([f"idx_{i}" for i in range(5)], name="index"),
        crs="EPSG:3067",
    )
    set_num_workers(workers)

    result = in_parallel(polygons).centerline(**kwargs).resolve_geoseries()

    assert result.crs == polygons.crs
    assert result.index.equals(polygons.index)
    expected = [centerline(polygon, **kwargs) for polygon in polygons]
    assert equals(result.to_numpy(), expected).all()


@pytest.mark.parametrize(
    "workers",
    [1, 3, 8],
    ids=["single_worker", "fewer_workers_than_geoms", "more_workers_than_geoms"],
)
@pytest.mark.parametrize(
    ("tree", "query_geoms", "kwargs", "expected"),
    [
        pytest.param(
            STRtree([box(i, 0, i + 1, 1) for i in range(5)]),
            [LineString([(0.5, 3), (3.5, 0.5)]), LineString([(4.2, 0.5), (4.8, 0.5)])],
            {},
            [[0, 0, 0, 0, 1], [0, 1, 2, 3, 4]],
            id="bounding_boxes",
        ),
        pytest.param(
            STRtree([box(i, 0, i + 1, 1) for i in range(5)]),
            [LineString([(0.5, 3), (3.5, 0.5)]), LineString([(4.2, 0.5), (4.8, 0.5)])],
            {"predicate": "intersects"},
            [[0, 0, 1], [2, 3, 4]],
            id="predicate",
        ),
        pytest.param(
            np.array([box(i, 0, i + 1, 1) for i in range(5)], dtype=object),
            [LineString([(0.5, 3), (3.5, 0.5)]), LineString([(4.2, 0.5), (4.8, 0.5)])],
            {"predicate": "intersects"},
            [[0, 0, 1], [2, 3, 4]],
            id="tree_built_from_geometries",
        ),
        pytest.param(
            STRtree([box(i, 0, i + 1, 1) for i in range(5)]),
            [Point(i, 3) for i in range(5)],
            {"predicate": "dwithin", "distance": 2.5},
            [
                [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
                [0, 1, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4],
            ],
            id="dwithin_scalar_distance",
        ),
        pytest.param(
            STRtree([box(i, 0, i + 1, 1) for i in range(5)]),
            [Point(i, 3) for i in range(5)],
            {"predicate": "dwithin", "distance": np.array([0.5, 2.5, 0.5, 2.5, 3])},
            [[1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4], [0, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4]],
            id="dwithin_distance_per_geometry",
        ),
        pytest.param(
            STRtree([box(i, 0, i + 1, 1) for i in range(5)]),
            [Point(100, 100), Point(200, 200)],
            {"predicate": "intersects"},
            [[], []],
            id="no_matches",
        ),
        pytest.param(
            STRtree([box(i, 0, i + 1, 1) for i in range(5)]),
            [],
            {"predicate": "intersects"},
            [[], []],
            id="empty",
        ),
    ],
)
def test_query_tree(
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    tree: STRtree | np.ndarray,
    query_geoms: list,
    kwargs: dict,
    expected: list[list[int]],
):
    set_num_workers(workers)
    monkeypatch.setattr(parallel, "_MIN_GEOMS_PER_QUERY_WORKER", 1)

    result = query_tree(tree, np.array(query_geoms, dtype=object), **kwargs)

    np.testing.assert_array_equal(result, expected)
