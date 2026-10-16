#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import Literal

import pytest
from geopandas.geodataframe import GeoDataFrame
from pytest_benchmark.fixture import BenchmarkFixture

from geogenalg.merge import (
    dissolve_and_inherit_attributes,
    merge_connecting_lines_by_attribute,
)
from geogenalg.testing import GeoPackagePath, TestInputData


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("layer", "expected_feature_count"),
    [
        ("100", 78),
        ("1000", 692),
        ("5000", 3500),
        ("10000", 7174),
    ],
    ids=[
        "100",
        "1000",
        "5000",
        "10000",
    ],
)
def test_merge_connecting_lines_by_attribute(
    benchmark: BenchmarkFixture,
    line_network_input: GeoPackagePath,
    layer: str,
    expected_feature_count: int,
):
    test_input = TestInputData(input_uri=line_network_input.to_input(layer))

    gdf, _, _ = test_input.read()

    result = benchmark(merge_connecting_lines_by_attribute, gdf, attribute="attribute")

    assert result.shape[0] == expected_feature_count


@pytest.mark.benchmark
@pytest.mark.parametrize("grouping_strategy", ["iterative", "bulk"])
@pytest.mark.parametrize(
    ("polygon_count"),
    [
        (100),
        (1000),
        (5000),
        (10000),
    ],
)
def test_dissolve_and_inherit_attributes_dense(
    benchmark: BenchmarkFixture,
    make_overlapping_polygons: Callable[..., GeoDataFrame],
    polygon_count: int,
    grouping_strategy: Literal["iterative", "bulk"],
):
    gdf = make_overlapping_polygons(polygon_count, spacing=0.1)

    result = benchmark(
        dissolve_and_inherit_attributes,
        gdf,
        "category",
        grouping_strategy=grouping_strategy,
    )

    assert sum(len(ids) for ids in result["old_ids"]) == polygon_count


@pytest.mark.benchmark
@pytest.mark.parametrize("grouping_strategy", ["iterative", "bulk"])
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
def test_dissolve_and_inherit_attributes_sparse(
    benchmark: BenchmarkFixture,
    make_overlapping_polygons: Callable[..., GeoDataFrame],
    polygon_count: int,
    grouping_strategy: Literal["iterative", "bulk"],
):
    gdf = make_overlapping_polygons(polygon_count, spacing=10)

    result = benchmark(
        dissolve_and_inherit_attributes,
        gdf,
        "category",
        grouping_strategy=grouping_strategy,
    )

    assert sum(len(ids) for ids in result["old_ids"]) == polygon_count
