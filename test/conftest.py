#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from warnings import warn

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_series_equal

from geogenalg.application import BaseAlgorithm
from geogenalg.core.exceptions import GeometryTypeError, MissingReferenceError
from geogenalg.testing import (
    AssertFunctionParameter,
    GeoPackageInput,
    TestGeoDataFrames,
    TestReportWarning,
    assert_gdf_equal_save_diff,
    get_test_gdfs,
)
from geogenalg.utility.validation import geometry_string_to_type


@pytest.fixture
def testdata_path() -> Path:
    return Path(__file__).resolve().parent / "testdata"


GEOMETRY_TYPE_STRINGS = (
    "Point",
    "LineString",
    "LinearRing",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
)


@dataclass(frozen=True)
class IntegrationTest:
    """Class for defining an integration test for a specific algorithm."""

    input_uri: GeoPackageInput | list[GeoPackageInput]
    """Path and layer for algorithm's input data."""
    control_uri: GeoPackageInput
    """Path and layer for test's control data."""
    algorithm: BaseAlgorithm
    """Algorithm instance."""
    unique_id_column: str
    """Name of column in input and reference data to set as GeoDataFrame index."""
    check_missing_reference: bool
    """If True, test will check that algorithm will raise a MissingReferenceError."""
    reference_uris: dict[str, GeoPackageInput] = field(default_factory=dict)
    """Paths and layers of algorithm's reference data."""
    assert_function_arguments: dict[AssertFunctionParameter, Any] = field(
        default_factory=dict
    )
    """Any arguments to pass to geopandas.testing.assert_geodataframe_equal."""

    def get_test_gdfs(self, *, geometry_column: str | None = None) -> TestGeoDataFrames:
        """Get GeoDataFrames used in test.

        Defined as a separate method for ease in debugging, allows inspecting
        results etc. without errors being raised.

        Returns:
        -------
            All GeoDataFrames used in the test.

        """

        return get_test_gdfs(
            input_uri=self.input_uri,
            control_uri=self.control_uri,
            alg=self.algorithm,
            unique_id_column=self.unique_id_column,
            reference_uris=self.reference_uris,
            rename_geometry=geometry_column,
        )

    def _assert_and_save_report(
        self,
        result: GeoDataFrame,
        control: GeoDataFrame,
    ) -> None:
        report_dir: Path | str | None = os.environ.get("GEOGENALG_TEST_REPORT_DIR")
        dir_specific = os.environ.get("GEOGENALG_TEST_REPORT_DIR_SPECIFIC")
        dir_is_specific = dir_specific is not None and dir_specific.lower() in {
            "true",
            "1",
            "on",
        }

        if report_dir is None:
            warn(
                "No directory specified for report. Using temporary directory by default. "
                + "You can set the GEOGENALG_TEST_REPORT_DIR environment variable to save to "
                + "a set location. By default report will be saved to a subdirectory according "
                + "to algorithm name and a timestamp. To save specifically to the set directory "
                + "and overwrite contents, set GEOGENALG_TEST_REPORT_DIR_SPECIFIC=true.",
                category=TestReportWarning,
                stacklevel=1,
            )
            report_dir = Path(gettempdir()) / "geogenalg_tests"
        else:
            report_dir = Path(report_dir)

        if not dir_is_specific:
            tz = datetime.now().astimezone().tzinfo
            timestamp = datetime.now(tz=tz).strftime("%Y_%m_%d-%H_%M_%S")
            prefix = self.algorithm.__class__.__name__ + "_"
            report_dir /= f"{prefix}{timestamp}"

        try:
            assert_gdf_equal_save_diff(
                result,
                control,
                assert_function_arguments=self.assert_function_arguments,
                directory=report_dir,
            )
        except:
            warn(
                "If the result is okay, you can make it the new control data by running: \n\n"
                + f"python tools/tests/write_layer.py {report_dir}/result.gpkg {self.control_uri.file}@{self.control_uri.layer_name}\n\n",
                category=TestReportWarning,
                stacklevel=1,
            )

            raise

    def run(self) -> None:
        """Run integration test."""
        input_data, input_data_before, _, result, control = self.get_test_gdfs()

        # Run this first so if report saving is on you can see the result (provided
        # no errors happen during algorithm execution).
        report_env = os.environ.get("GEOGENALG_TEST_REPORT_SAVE")
        save_report = report_env is not None and report_env.lower() in {
            "true",
            "1",
            "on",
        }

        if not save_report:
            assert_geodataframe_equal(
                result,
                control,
                **self.assert_function_arguments,
            )
            # Test GeoSeries separately, because assert_geodataframe_equal
            # does not check that Z values are equal.
            assert_series_equal(result.geometry, control.geometry)
        else:
            self._assert_and_save_report(result, control)

        assert input_data.crs == result.crs
        assert input_data.crs == control.crs

        # Ensure input data was not modified.
        # TODO: we should probably ensure that reference data is also unmodified
        assert_geodataframe_equal(input_data, input_data_before)

        self._check_test_data_has_z_coordinates(input_data, control, result)
        self._check_test_data_single_geometries(input_data, result)

        if self.check_missing_reference:
            with pytest.raises(
                MissingReferenceError, match=r"Reference data is missing."
            ):
                self.algorithm.execute(input_data)

        for string in GEOMETRY_TYPE_STRINGS:
            if string in self.algorithm.valid_input_geometry_types:
                continue

            geom_type = geometry_string_to_type(string)

            with pytest.raises(GeometryTypeError):
                self.algorithm.execute(
                    GeoDataFrame(geometry=[geom_type()]),
                )

        # TODO: test reference data geom types?

        # By default run algorithm twice with same data, to test that it
        # produces same results with different geometry column names. Allow
        # disabling this by environment variable to speed up tests during
        # development.
        geom_column = os.environ.get("GEOGENALG_TEST_ONE_GEOM_COLUMN")
        test_geom_column = geom_column is None or geom_column.lower() not in {
            "true",
            "1",
            "on",
        }

        if test_geom_column:
            _, _, _, result_from_geom_column, control_from_geom_column = (
                self.get_test_gdfs(geometry_column="geom")
            )
            if not save_report:
                assert_geodataframe_equal(
                    result_from_geom_column,
                    control_from_geom_column,
                    **self.assert_function_arguments,
                )
            else:
                self._assert_and_save_report(
                    result_from_geom_column, control_from_geom_column
                )

    def _check_test_data_has_z_coordinates(
        self,
        input_data: GeoDataFrame,
        control: GeoDataFrame,
        result: GeoDataFrame,
    ) -> None:
        """Check that test data and results uniformly have z coordinates.

        Raises
        ------
            AssertionError: If a check fails.

        """
        if not input_data.has_z.all():
            msg = "Input data for integation test must have geometries with Z values."
            raise AssertionError(msg)

        if not control.has_z.all():
            msg = "Control data for integation test must have geometries with Z values."
            raise AssertionError(msg)

        if input_data.has_z.any() and not input_data.has_z.all():
            msg = "Input data has mixed 2.5D and 2D geometries."
            raise AssertionError(msg)

        if result.has_z.any() and not result.has_z.all():
            msg = "Result has mixed 2.5D and 2D geometries."
            raise AssertionError(msg)

        if control.has_z.any() and not control.has_z.all():
            msg = "Control data has mixed 2.5D and 2D geometries."
            raise AssertionError(msg)

        if control.has_z.all() != result.has_z.all():
            msg = "Control or result data has z when other does not."
            raise AssertionError(msg)

    def _check_test_data_single_geometries(
        self,
        input_data: GeoDataFrame,
        result: GeoDataFrame,
    ) -> None:
        """Check that if input data has only single geometries, result does also.

        Raises
        ------
            AssertionError: If a check fails.

        """
        input_has_only_single_geometries = all(
            "Multi" not in geom_type
            for geom_type in input_data.geometry.geom_type.to_numpy()
        )
        result_has_only_single_geometries = all(
            "Multi" not in geom_type
            for geom_type in result.geometry.geom_type.to_numpy()
        )

        if input_has_only_single_geometries and not result_has_only_single_geometries:
            msg = "Input has only single geometries but result does not."
            raise AssertionError(msg)
