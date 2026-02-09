#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

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


# TODO: Once GeneralizeLandcover supports identity, refactor to something like this:
# def test_generalize_landcover(
#     testdata_path: Path,
# ) -> None:
#     gpkg = GeoPackagePath(testdata_path / "marshes_small_area.gpkg")  # noqa: ERA001
#
#     IntegrationTest(
#         input_uri=gpkg.to_input("mtk_marshes"),  # noqa: ERA001
#         control_uri=gpkg.to_input("generalized_marshes"),  # noqa: ERA001
#         algorithm=GeneralizeLandcover(  # noqa: ERA001, RUF100
#             buffer_constant=20,  # noqa: ERA001
#             simplification_tolerance=30,  # noqa: ERA001
#             area_threshold=7500,  # noqa: ERA001
#             hole_threshold=5000,  # noqa: ERA001
#             smoothing=True,  # noqa: ERA001
#         ),
#         unique_id_column=UNIQUE_ID_COLUMN,  # noqa: ERA001
#         check_missing_reference=False,  # noqa: ERA001
#     ).run()


def test_generalize_landcover_50k(
    testdata_path: Path,
) -> None:
    """
    Test generalizing landcover with marshes
    """
    input_path = testdata_path / "marshes_2.gpkg"
    input_data = read_gdf_from_file_and_set_index(
        input_path,
        UNIQUE_ID_COLUMN,
        layer="marsh",
    )

    temp_dir = TemporaryDirectory()
    output_path = Path(temp_dir.name) / "generalized_marshes.gpkg"

    algorithm = GeneralizeLandcover(
        positive_buffer=25,
        negative_buffer=-10,
        simplification_tolerance=15,
        area_threshold=5000,
        hole_threshold=5000,
        smoothing=True,
    )
    algorithm.execute(input_data, {}).to_file(output_path, layer="marshes_50k")
    result_marshes = read_gdf_from_file_and_set_index(
        output_path, UNIQUE_ID_COLUMN, layer="marshes_50k"
    )

    control_marshes: GeoDataFrame = read_gdf_from_file_and_set_index(
        input_path, UNIQUE_ID_COLUMN, layer="generalized_marsh"
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
        positive_buffer=25,
        negative_buffer=-10,
        simplification_tolerance=15,
        area_threshold=5000,
        hole_threshold=5000,
        smoothing=True,
    )

    with pytest.raises(
        GeometryTypeError,
        match=r"Input data must contain only geometries of following types: Polygon.",
    ):
        algorithm.execute(data=gdf, reference_data={})
