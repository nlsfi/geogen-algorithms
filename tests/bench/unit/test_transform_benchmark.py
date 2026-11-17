#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from geogenalg.testing import TestInputData
from geogenalg.transform import boffet_areas


@pytest.mark.benchmark
def test_boffet_areas(
    benchmark: BenchmarkFixture,
    building_areas_input: TestInputData,
):
    gdf, _, _ = building_areas_input.read()

    result = benchmark(boffet_areas, gdf.geometry, 10, 10)

    assert result.shape[0] == 112
