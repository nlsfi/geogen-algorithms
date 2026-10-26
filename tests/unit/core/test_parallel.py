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
    box,
    buffer,
    equals,
    length,
)
from shapely.geometry import Point

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
    set_num_workers,
    shutdown_threadpool_executor,
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
