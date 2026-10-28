#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


import numpy as np
import pytest
from geopandas import GeoDataFrame, GeoSeries
from shapely import buffer, points
from shapely.geometry import LineString, box


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int | pytest.ExitCode,
    config: pytest.Config,
):
    passed = terminalreporter.stats.get("passed", []), {"green": True}
    failed = terminalreporter.stats.get("failed", []), {"red": True}

    if not passed[0] and not failed[0]:
        return

    terminalreporter.section(
        "Algorithm Throughput Report",
        sep="-",
        cyan=True,
    )

    for reports, kwargs in (passed, failed):
        for report in reports:
            for key, value in getattr(report, "user_properties", []):
                if key == "features_per_sec":
                    test_name = report.nodeid.split("::")[-1]
                    terminalreporter.write_line(
                        f"{test_name:<40} {value:>12} features/sec", **kwargs
                    )

    terminalreporter.write_sep("-", cyan=True)


@pytest.fixture
def make_linestring():
    def _factory(vertex_count: int, seed: int = 0) -> LineString:
        rng = np.random.default_rng(seed=seed)

        steps = rng.normal(loc=1.0, scale=0.5, size=(vertex_count, 2))
        coords = np.cumsum(steps, axis=0)

        return LineString(coords)

    return _factory


@pytest.fixture
def make_point_gdf():
    def _factory(
        point_count: int,
        seed: int = 0,
        spacing: float = 2.0,
        has_z: bool = False,
    ) -> GeoDataFrame:
        rng = np.random.default_rng(seed=seed)

        extent = spacing * np.sqrt(point_count)
        coords = rng.uniform(0.0, extent, size=(point_count, 2))

        if has_z:
            z = rng.uniform(10.0, 500.0, size=(point_count, 1))
            coords = np.hstack([coords, z])

        geometries = points(coords)

        return GeoDataFrame(
            {"category": rng.integers(0, 3, size=point_count)},
            geometry=geometries,
            crs="EPSG:3857",
        )

    return _factory


@pytest.fixture
def make_overlapping_polygons():
    def _factory(
        polygon_count: int,
        seed: int = 0,
        spacing: float = 2.0,
    ) -> GeoDataFrame:
        rng = np.random.default_rng(seed=seed)

        extent = spacing * np.sqrt(polygon_count)
        centers = rng.uniform(0.0, extent, size=(polygon_count, 2))
        radii = rng.uniform(0.5, 1.5, size=polygon_count)

        geometries = buffer(points(centers), radii)

        return GeoDataFrame(
            {"category": rng.integers(0, 3, size=polygon_count)},
            geometry=geometries,
            crs="EPSG:3857",
        )

    return _factory


@pytest.fixture
def make_adjacent_polygons_with_boundary_lines():
    def _factory(polygon_count: int) -> tuple[GeoDataFrame, GeoDataFrame]:
        polygons = [box(i, 0, i + 1, 1) for i in range(polygon_count)]
        lines = [
            LineString([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)])
            for i in range(polygon_count)
        ]

        return (
            GeoDataFrame(geometry=polygons, crs="EPSG:3857"),
            GeoDataFrame(geometry=lines, crs="EPSG:3857"),
        )

    return _factory


@pytest.fixture
def make_polygon_grid():
    def _factory(grid_size: int, vertices_per_edge: int) -> GeoDataFrame:
        return GeoDataFrame(
            geometry=GeoSeries(
                [
                    box(x, y, x + 1, y + 1).segmentize(1 / vertices_per_edge)
                    for x in range(grid_size)
                    for y in range(grid_size)
                ],
                crs="EPSG:3857",
            ),
        )

    return _factory
