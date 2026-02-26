#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import pytest
from geopandas import GeoDataFrame, read_file
from geopandas.testing import assert_geodataframe_equal
from pandas import concat
from shapely import Point, Polygon, box

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.testing import (
    GeoPackageInput,
    assert_gdf_equal_save_diff,
    get_alg_results_from_geopackage,
    get_test_gdfs,
)


def test_assert_gdf_equal_save_diff_geom():
    temp_dir = TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    gdf_result = GeoDataFrame(
        geometry=[box(0.5, 0.5, 1.5, 1.5)],
        crs="EPSG:3857",
    )
    gdf_control = GeoDataFrame(
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:3857",
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
        crs="EPSG:3857",
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


@supports_identity
@dataclass(frozen=True)
class MockAlg(BaseAlgorithm):
    mock_attribute: str = "test"

    valid_input_geometry_types: ClassVar = {"Point"}
    valid_reference_geometry_types: ClassVar = {"Point"}

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        result = GeoDataFrame(
            {"id": ["3"], "column": [self.mock_attribute]},
            geometry=[Point(2, 0)],
            crs="EPSG:3067",
        )
        result = result.set_index("id")

        return GeoDataFrame(
            concat(
                [
                    data,
                    reference_data["ref"],
                    result,
                ]
            )
        )


def test_get_alg_results_from_geopackage():
    input_data = GeoDataFrame(
        {"id": ["1"], "column": ["input"]},
        geometry=[Point(0, 0)],
        crs="EPSG:3067",
    )
    ref_data = GeoDataFrame(
        {"id": ["2"], "column": ["ref_data"]},
        geometry=[Point(1, 0)],
        crs="EPSG:3067",
    )

    input_data = input_data.set_index("id")
    ref_data = ref_data.set_index("id")

    result = get_alg_results_from_geopackage(
        MockAlg(mock_attribute="should_not_be_test"),
        input_data,
        "id",
        {"ref": ref_data},
    )

    control = GeoDataFrame(
        {
            "id": ["1", "2", "3"],
            "column": ["input", "ref_data", "should_not_be_test"],
        },
        geometry=[
            Point(0, 0),
            Point(1, 0),
            Point(2, 0),
        ],
        crs="EPSG:3067",
    )
    control = control.set_index("id")

    assert_geodataframe_equal(result, control)


def test_get_result_and_control():
    input_data = GeoDataFrame(
        {"id": ["1"], "column": ["input"]},
        geometry=[Point(0, 0)],
        crs="EPSG:3067",
    )
    ref_data = GeoDataFrame(
        {"id": ["2"], "column": ["ref_data"]},
        geometry=[Point(1, 0)],
        crs="EPSG:3067",
    )
    control_data = GeoDataFrame(
        {"id": ["5"], "column": ["control"]},
        geometry=[Point(1, 0)],
        crs="EPSG:3067",
    )

    temp_dir = TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    input_path = GeoPackageInput(temp_dir_path / "input.gpkg", layer_name="input")
    ref_path = GeoPackageInput(temp_dir_path / "ref.gpkg", layer_name="ref")
    control_path = GeoPackageInput(temp_dir_path / "control.gpkg", layer_name="control")

    input_data.to_file(input_path.file, layer=input_path.layer_name)
    ref_data.to_file(ref_path.file, layer=ref_path.layer_name)
    control_data.to_file(control_path.file, layer=control_path.layer_name)

    other_input, input_before, other_ref, result, control = get_test_gdfs(
        input_path,
        control_path,
        MockAlg("result"),
        "id",
        reference_uris={"ref": ref_path},
    )

    assert_geodataframe_equal(input_data.set_index("id"), input_before)
    assert_geodataframe_equal(input_data.set_index("id"), other_input)
    assert_geodataframe_equal(ref_data.set_index("id"), other_ref["ref"])

    result_expected = GeoDataFrame(
        {
            "id": ["1", "2", "3"],
            "column": ["input", "ref_data", "result"],
        },
        geometry=[
            Point(0, 0),
            Point(1, 0),
            Point(2, 0),
        ],
        crs="EPSG:3067",
    )
    result_expected = result_expected.set_index("id")

    control_expected = control_data.copy()
    control_expected = control_expected.set_index("id")

    assert_geodataframe_equal(result, result_expected)
    assert_geodataframe_equal(control, control_expected)
