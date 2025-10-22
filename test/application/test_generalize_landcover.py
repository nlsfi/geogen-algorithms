#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from geopandas import GeoDataFrame, read_file
from geopandas.testing import assert_geodataframe_equal
from shapely import Point

from geogenalg.application.generalize_landcover import GeneralizeLandcover
from geogenalg.core.exceptions import GeometryTypeError


def _instantiate_algorithm() -> GeneralizeLandcover:
    return GeneralizeLandcover(
        buffer_constant=20,
        simplification_tolerance=30,
        area_threshold=7500,
        hole_threshold=5000,
        smoothing=True,
    )


def test_generalize_landcover_50k(
    testdata_path: Path,
) -> None:
    """
    Test generalizing landcover with cultivated land class
    """
    source_path = testdata_path / "marshes_small_area.gpkg"

    temp_dir = TemporaryDirectory()
    output_path = temp_dir.name + "/generalized_marshes.gpkg"

    input_gdf = read_file(source_path, layer="mtk_marshes")

    algorithm = _instantiate_algorithm()

    result = algorithm.execute(input_gdf, {})

    assert result is not None

    result.to_file(output_path, layer="marshes_50k")

    control_marshes: GeoDataFrame = read_file(source_path, layer="generalized_marshes")

    result_marshes = read_file(output_path, layer="marshes_50k")

    control_marshes = control_marshes.sort_values("geometry").reset_index(drop=True)
    result_marshes = result_marshes.sort_values("geometry").reset_index(drop=True)

    # TODO: figure out how to pass Geojson geometries without rounding errors
    assert_geodataframe_equal(
        control_marshes.drop(columns=["sijainti_piste"]),
        result_marshes.drop(columns=["sijainti_piste"]),
        check_index_type=False,
    )


def test_generalize_landcover_invalid_geometry_type() -> None:
    gdf = GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])

    algorithm = _instantiate_algorithm()

    with pytest.raises(
        GeometryTypeError,
        match=r"GeneralizeLandcover works only with Polygon geometries.",
    ):
        algorithm.execute(data=gdf, reference_data={})
