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

from geogenalg.application import BaseAlgorithm
from geogenalg.core.exceptions import GeometryTypeError, MissingReferenceError
from geogenalg.testing import (
    AssertFunctionParameter,
    DiffWarning,
    GeoPackageInput,
    TestGeoDataFrames,
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

    def _assert_and_save_diffs(
        self,
        result: GeoDataFrame,
        control: GeoDataFrame,
    ) -> None:
        diff_dir: Path | str | None = os.environ.get("GEOGENALG_TEST_DIFF_DIR")
        dir_specific = os.environ.get("GEOGENALG_TEST_DIFF_DIR_SPECIFIC")
        dir_is_specific = dir_specific is not None and dir_specific.lower() in {
            "true",
            "1",
            "on",
        }

        if diff_dir is None:
            warn(
                "No directory specified for diff. Using temporary directory by default. "
                + "You can set the GEOGENALG_TEST_DIFF_DIR environment variable to save to "
                + "a set location. By default diffs will be saved to a subdirectory according "
                + "to algorithm name and a timestamp. To save specifically to the set directory "
                + "and overwrite contents, set GEOGENALG_TEST_DIFF_DIR_SPECIFIC=true.",
                category=DiffWarning,
                stacklevel=1,
            )
            diff_dir = Path(gettempdir()) / "geogenalg_tests"
        else:
            diff_dir = Path(diff_dir)

        if not dir_is_specific:
            tz = datetime.now().astimezone().tzinfo
            timestamp = datetime.now(tz=tz).strftime("%Y_%m_%d-%H_%M_%S")
            prefix = self.algorithm.__class__.__name__ + "_"
            diff_dir /= f"{prefix}{timestamp}"

        assert_gdf_equal_save_diff(
            result,
            control,
            assert_function_arguments=self.assert_function_arguments,
            directory=diff_dir,
        )

    def run(self) -> None:  # noqa: C901, PLR0912
        """Run integration test.

        Raises
        ------
            AssertionError: If test fails.

        """
        input_data, input_data_before, _, result, control = self.get_test_gdfs()

        assert input_data.crs == result.crs
        assert input_data.crs == control.crs

        # Ensure input data was not modified.
        # TODO: we should probably ensure that reference data is also unmodified
        assert_geodataframe_equal(input_data, input_data_before)

        # TODO: change these to failure states? However, that would
        # require changes to some algorithms, so only warn for now.
        if not input_data.has_z.all():
            warn(
                f"{type(self.algorithm).__name__}: Input data should have only geometries with z included.",
                stacklevel=2,
            )
            if not control.has_z.all():
                warn(
                    f"{type(self.algorithm).__name__}: Control data should have only geometries with z included.",
                    stacklevel=2,
                )

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

        input_has_only_single_geometries = all(
            "Multi" not in geom_type
            for geom_type in input_data.geometry.geom_type.values
        )
        result_has_only_single_geometries = all(
            "Multi" not in geom_type for geom_type in result.geometry.geom_type.values
        )

        if input_has_only_single_geometries and not result_has_only_single_geometries:
            msg = "Input has only single geometries but result does not."
            raise AssertionError(msg)

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

        diff_env = os.environ.get("GEOGENALG_TEST_SAVE_DIFF")
        save_diff = diff_env is not None and diff_env.lower() in {
            "true",
            "1",
            "on",
        }

        if not save_diff:
            assert_geodataframe_equal(
                result,
                control,
                **self.assert_function_arguments,
            )
        else:
            self._assert_and_save_diffs(result, control)

        # By default run algorithm twice with same data, to test that it
        # produces same results with different geometry column names. Allow
        # disabling this by environment variable to speed up tests during
        # development.
        geom_column = os.environ.get("GEOGENALG_TEST_NO_GEOM_COLUMNS")
        test_geom_column = geom_column is None or geom_column.lower() not in {
            "true",
            "1",
            "on",
        }

        if test_geom_column:
            _, _, _, result_from_geom_column, control_from_geom_column = (
                self.get_test_gdfs(geometry_column="geom")
            )
            if not save_diff:
                assert_geodataframe_equal(
                    result_from_geom_column,
                    control_from_geom_column,
                    **self.assert_function_arguments,
                )
            else:
                self._assert_and_save_diffs(
                    result_from_geom_column, control_from_geom_column
                )
