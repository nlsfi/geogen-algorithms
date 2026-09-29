#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from geogenalg.continuity import (
    add_contiguous_lines_information,
    count_connections,
    flag_connections,
    flag_connections_to_reference,
)
from geogenalg.testing import GeoPackagePath, TestInputData


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("layer", "start_sum", "end_sum"),
    [
        ("100", 308, 160),
        ("1000", 4339, 2397),
        ("5000", 33692, 20212),
        ("10000", 86091, 49821),
    ],
    ids=[
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
    start_sum: int,
    end_sum: int,
):
    test_input = TestInputData(input_uri=line_network_input.to_input(layer))

    gdf, _, _ = test_input.read()

    result = benchmark(count_connections, gdf)

    assert result["_start_connections"].sum() == start_sum
    assert result["_end_connections"].sum() == end_sum


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("layer", "start_sum", "end_sum"),
    [
        ("100", 98, 61),
        ("1000", 980, 778),
        ("5000", 4900, 4590),
        ("10000", 9800, 9495),
    ],
    ids=[
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
    start_sum: int,
    end_sum: int,
):
    test_input = TestInputData(input_uri=line_network_input.to_input(layer))

    gdf, _, _ = test_input.read()

    result = benchmark(flag_connections, gdf)

    assert result["_start_connected"].sum() == start_sum
    assert result["_end_connected"].sum() == end_sum


@pytest.mark.benchmark
def test_flag_connections_to_reference(
    benchmark: BenchmarkFixture,
    line_network_with_reference_input: GeoPackagePath,
):
    test_input = TestInputData(
        input_uri=line_network_with_reference_input.to_input("10000"),
    )

    gdf, _, _ = test_input.read()
    in_gdf = gdf.loc[gdf["group"] == 1]
    ref_gdf = gdf.loc[gdf["group"] == 2]

    result = benchmark(
        flag_connections_to_reference,
        input_gdf=in_gdf,
        reference_gdf=ref_gdf,
    )

    assert result["_start_connected"].sum() == 4485
    assert result["_end_connected"].sum() == 4113


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("layer", "deadend_sum", "disconnected_sum"),
    [
        ("100", 39, 2),
        ("1000", 236, 20),
        ("5000", 315, 100),
        ("10000", 314, 200),
    ],
    ids=[
        "100",
        "1000",
        "5000",
        "10000",
    ],
)
def test_add_contiguous_lines_information(
    benchmark: BenchmarkFixture,
    line_network_input: GeoPackagePath,
    layer: str,
    deadend_sum: int,
    disconnected_sum: int,
):
    test_input = TestInputData(input_uri=line_network_input.to_input(layer))

    gdf, _, _ = test_input.read()

    result = benchmark(add_contiguous_lines_information, gdf)

    assert result["contiguous_dead_end"].sum() == deadend_sum
    assert result["contiguous_disconnected"].sum() == disconnected_sum
