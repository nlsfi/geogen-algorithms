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
from shapely import Point, Polygon, box

from geogenalg.testing import assert_gdf_equal_save_diff


def test_assert_gdf_equal_save_diff_geom():
    temp_dir = TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    gdf_result = GeoDataFrame(
        geometry=[box(0.5, 0.5, 1.5, 1.5)],
    )
    gdf_control = GeoDataFrame(
        geometry=[box(0, 0, 1, 1)],
    )

    with pytest.raises(AssertionError):
        assert_gdf_equal_save_diff(gdf_result, gdf_control, directory=temp_dir_path)

    result_path = temp_dir_path / "result.gpkg"
    geomdiff_path = temp_dir_path / "geomdiff.gpkg"
    control_path = temp_dir_path / "control.gpkg"

    assert result_path.exists()
    assert geomdiff_path.exists()
    assert control_path.exists()

    control = read_file(control_path)
    result = read_file(result_path)
    assert_geodataframe_equal(control, gdf_control)
    assert_geodataframe_equal(result, gdf_result)

    geomdiff = read_file(geomdiff_path)
    geomdiff_expected = GeoDataFrame(
        geometry=[
            Polygon(
                [
                    [0.5, 1.5],
                    [1.5, 1.5],
                    [1.5, 0.5],
                    [1.0, 0.5],
                    [1.0, 1.0],
                    [0.5, 1.0],
                    [0.5, 1.5],
                ]
            )
        ],
    )
    assert_geodataframe_equal(geomdiff, geomdiff_expected)


def test_assert_gdf_equal_save_diff_is_equal():
    temp_dir = TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    gdf1 = GeoDataFrame(
        {id: [1, 2]},
        geometry=[Point(0, 0), Point(1, 0)],
    )

    gdf2 = gdf1.copy()

    assert_gdf_equal_save_diff(gdf1, gdf2, directory=temp_dir_path)

    assert not (temp_dir_path / "result.gpkg").exists()
    assert not (temp_dir_path / "geomdiff.gpkg").exists()
    assert not (temp_dir_path / "control.gpkg").exists()
