#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from shapely import Point

from geogenalg.application.generalize_landcover import GeneralizeLandcover
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

UNIQUE_ID_COLUMN = "mtk_id"


def test_generalize_landcover_50k(
    testdata_path: Path,
) -> None:
    """
    Test generalizing landcover with cultivated land class
    """
    input_path = testdata_path / "marshes_small_area.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="mtk_marshes",
    )

    temp_dir = TemporaryDirectory()
    output_path = Path(temp_dir.name) / "generalized_marshes.gpkg"

    algorithm = GeneralizeLandcover(
        positive_buffer=20,
        negative_buffer=-20,
        simplification_tolerance=30,
        area_threshold=7500,
        hole_threshold=5000,
        smoothing=True,
    )
    algorithm.execute(input_data, {}).to_file(output_path, layer="marshes_50k")
    result_marshes = read_gdf_from_file_and_set_index(
        output_path, "index", layer="marshes_50k"
    )

    control_marshes: GeoDataFrame = read_gdf_from_file_and_set_index(
        input_path, "index", layer="generalized_marshes"
    )

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

    algorithm = GeneralizeLandcover(
        positive_buffer=20,
        negative_buffer=-20,
        simplification_tolerance=30,
        area_threshold=7500,
        hole_threshold=5000,
        smoothing=True,
    )

    with pytest.raises(
        GeometryTypeError,
        match=r"GeneralizeLandcover works only with Polygon geometries.",
    ):
        algorithm.execute(data=gdf, reference_data={})
