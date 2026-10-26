#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from __future__ import annotations

import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

import numpy as np
from geopandas import GeoSeries
from geopandas.array import GeometryArray
from pandas import Index, Series
from shapely import (
    BufferCapStyle,
    BufferJoinStyle,
    buffer,
    difference,
    intersection,
    symmetric_difference,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyproj import CRS
    from shapely.geometry.base import BaseGeometry

GEOGENALG_WORKER_COUNT_ENV_VAR = "GEOGENALG_NUM_WORKERS"


class _ThreadManager:
    """Private singleton class for managing a thread pool executor."""

    def __init__(self) -> None:
        self._num_workers: int | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._mutex = threading.Lock()

        atexit.register(self.shutdown)

    def get_worker_count(self) -> int:
        # Prefer explicitly set number
        if self._num_workers is not None:
            return self._num_workers

        # Env variable as a first fallback
        if (env_variable := os.getenv(GEOGENALG_WORKER_COUNT_ENV_VAR)) is not None:
            try:
                as_int = int(env_variable)
                if as_int > 0:
                    return as_int
                warn(
                    f"{GEOGENALG_WORKER_COUNT_ENV_VAR} must be > 0. Using default.",
                    UserWarning,
                    stacklevel=3,
                )
            except ValueError:
                warn(
                    f"{GEOGENALG_WORKER_COUNT_ENV_VAR} could not be parsed as an "
                    "integer. Using default.",
                    UserWarning,
                    stacklevel=3,
                )
        # No explicit setting or env variable -> try to use number of cores

        # Only in >= Python 3.13
        if hasattr(os, "process_cpu_count") and (count := os.process_cpu_count()):
            return count

        # For Python 3.12 (UNIX only)
        if hasattr(os, "sched_getaffinity"):
            try:
                return len(os.sched_getaffinity(0))
            except OSError:
                pass

        return os.cpu_count() or 1

    @property
    def executor(self) -> ThreadPoolExecutor:
        with self._mutex:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.get_worker_count())

            return self._executor

    @property
    def num_workers(self) -> int:
        return self.get_worker_count()

    @num_workers.setter
    def num_workers(self, value: int | None) -> None:
        if value is not None and value < 1:
            msg = "Value must be >= 1."
            raise ValueError(msg)

        with self._mutex:
            self._num_workers = value

            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = ThreadPoolExecutor(max_workers=self.get_worker_count())

    def shutdown(self) -> None:
        with self._mutex:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None


_THREAD_MANAGER = _ThreadManager()


def num_workers() -> int:
    """Return number of workers."""
    return _THREAD_MANAGER.num_workers


def set_num_workers(value: int | None) -> None:
    """Set number of workers and reconfigure thread pool."""
    _THREAD_MANAGER.num_workers = value


def get_threadpool_executor() -> ThreadPoolExecutor:
    """Get thread pool executor."""
    return _THREAD_MANAGER.executor


def shutdown_threadpool_executor() -> None:
    """Manually shutdown thread pool executor."""
    _THREAD_MANAGER.shutdown()


def _get_chunk_slices(value_count: int, chunk_count: int) -> list[slice]:
    chunk_size, remainder = divmod(value_count, chunk_count)
    stops = [i * chunk_size + min(i, remainder) for i in range(chunk_count + 1)]
    return [slice(stops[i], stops[i + 1]) for i in range(chunk_count)]


def _process_arg(
    value: object,
    _slice: slice,
    total_geom_count: int,
) -> slice | object:
    if isinstance(value, np.ndarray) and len(value) == total_geom_count:
        return value[_slice]

    if isinstance(value, Series) and len(value) == total_geom_count:
        return value.iloc[_slice]

    return value


def _process_geometry_array(
    geoms: np.ndarray,
    geom_function: Callable[..., Any],
    *args: object,
    executor: ThreadPoolExecutor | None = None,
    **kwargs: object,
) -> np.ndarray:
    geom_count = len(geoms)
    if geom_count == 0:
        return np.array([], dtype="object")

    workers = min(num_workers(), geom_count)
    if workers <= 1:
        return np.asarray(geom_function(geoms, *args, **kwargs))

    def worker(_slice: slice) -> np.ndarray:
        worker_args = [_process_arg(arg, _slice, geom_count) for arg in args]
        worker_kwargs = {
            key: _process_arg(value, _slice, geom_count)
            for key, value in kwargs.items()
        }
        return np.asarray(
            geom_function(geoms[_slice], *worker_args, **worker_kwargs),
        )

    pool = executor or get_threadpool_executor()
    return np.concatenate(
        list(pool.map(worker, _get_chunk_slices(geom_count, workers))),
    )


class ParallelGeometryProcess:
    """Convenience class to allow chaining geometry operations.

    All operations are are executed in parallel (depending on the number of
    workers configured.)

    Objects are intended to be instantiated with the in_parallel() function.
    """

    def __init__(
        self,
        data: np.ndarray,
        index: Index,
        crs: str | int | tuple[str, str] | list[str] | CRS | None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        """Initialize object."""
        self._data = data
        self._index = index
        self._crs = crs
        self._executor = executor

    def _run(
        self,
        func: Callable[..., Any],
        *args: object,
        **kwargs: object,
    ) -> ParallelGeometryProcess:
        """Run geometry processing function in parallel."""
        processed_args = [
            a.to_numpy() if isinstance(a, (Series, GeoSeries)) else a for a in args
        ]

        processed_kwargs = {
            k: (v.to_numpy() if isinstance(v, (Series, GeoSeries)) else v)
            for k, v in kwargs.items()
        }

        result = _process_geometry_array(
            self._data,
            func,
            *processed_args,
            executor=self._executor,
            **processed_kwargs,
        )

        return ParallelGeometryProcess(
            result,
            self._index,
            self._crs,
            self._executor,
        )

    def buffer(  # noqa: PLR0913
        self,
        distance: float | np.ndarray | Series,
        resolution: int = 16,
        cap_style: BufferCapStyle | Literal["round", "square", "flat"] = "round",
        join_style: BufferJoinStyle | Literal["round", "mitre", "bevel"] = "round",
        mitre_limit: float = 5.0,
        *,
        single_sided: bool = False,
    ) -> ParallelGeometryProcess:
        """Run shapely buffer in parallel."""
        return self._run(
            buffer,
            distance,
            quad_segs=resolution,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided,
        )

    def intersection(
        self,
        other: BaseGeometry | np.ndarray | GeoSeries,
        grid_size: float | None = None,
    ) -> ParallelGeometryProcess:
        """Run shapely intersection in parallel."""
        return self._run(intersection, other, grid_size=grid_size)

    def difference(
        self,
        other: BaseGeometry | np.ndarray | GeoSeries,
        grid_size: float | None = None,
    ) -> ParallelGeometryProcess:
        """Run shapely difference in parallel."""
        return self._run(difference, other, grid_size=grid_size)

    def symmetric_difference(
        self,
        other: BaseGeometry | np.ndarray | GeoSeries,
        grid_size: float | None = None,
    ) -> ParallelGeometryProcess:
        """Run shapely symmetric difference in parallel."""
        return self._run(symmetric_difference, other, grid_size=grid_size)

    def resolve_series(self) -> Series:
        """Turn to series."""
        return Series(self._data, index=self._index)

    def resolve_geoseries(self) -> GeoSeries:
        """Turn to geoseries."""
        return GeoSeries(
            GeometryArray(self._data, crs=self._crs),
            index=self._index,
        )


def in_parallel(
    geoms: GeoSeries,
    executor: ThreadPoolExecutor | None = None,
) -> ParallelGeometryProcess:
    """Create geometry process object from GeoSeries."""
    return ParallelGeometryProcess(
        data=geoms.to_numpy(),
        index=geoms.index,
        crs=geoms.crs,
        executor=executor,
    )
