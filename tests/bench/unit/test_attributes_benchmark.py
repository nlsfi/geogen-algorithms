#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
from geopandas import GeoDataFrame
from pytest_benchmark.fixture import BenchmarkFixture
from shapely import (
    boundary,
    get_parts,
    get_segments,  # ty: ignore[unresolved-import] (type stubs are for shapely 2.1)
)

from geogenalg.attributes import inherit_attributes_from_largest
from geogenalg.testing import TestInputData


@pytest.mark.parametrize("step", [2, 10, 50])
@pytest.mark.benchmark
def test_inherit_attributes_from_largest(
    step: int,
    benchmark: BenchmarkFixture,
    watercourse_areas_input: TestInputData,
):
    polygons, _, _ = watercourse_areas_input.read()
    polygons = polygons.loc[polygons.geometry.geom_type.isin(["Polygon"])]
    segments = get_segments(get_parts(boundary(polygons.geometry.to_numpy())))
    targets = GeoDataFrame(geometry=segments[::step], crs=polygons.crs)

    result = benchmark(
        inherit_attributes_from_largest,
        polygons,
        targets,
        old_ids_column="_old_ids",
    )

    assert len(result) == len(targets)
