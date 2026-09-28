#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from geogenalg.continuity import count_connections, flag_connections
from geogenalg.testing import GeoPackagePath, TestInputData


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "layer",
    [
        "100",
        "1000",
        "5000",
        "10000",
    ],
)
def test_count_connections(
    benchmark: BenchmarkFixture,
    line_network_input: GeoPackagePath,
    layer: str,
):
    test_input = TestInputData(input_uri=line_network_input.to_input(layer))

    gdf, _, _ = test_input.read()

    benchmark(count_connections, gdf)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "layer",
    [
        "100",
        "1000",
        "5000",
        "10000",
    ],
)
def test_flag_connections(
    benchmark: BenchmarkFixture,
    line_network_input: GeoPackagePath,
    layer: str,
):
    test_input = TestInputData(input_uri=line_network_input.to_input(layer))

    gdf, _, _ = test_input.read()

    benchmark(flag_connections, gdf)
