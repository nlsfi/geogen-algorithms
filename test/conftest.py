#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from warnings import warn

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal

from geogenalg.application import BaseAlgorithm
from geogenalg.core.exceptions import GeometryTypeError, MissingReferenceError
from geogenalg.testing import GeoPackageInput, TestGeoDataFrames, get_test_gdfs
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
    assert_function_kwargs: dict[str, Any] = field(default_factory=dict)
    """Any arguments to pass to geopandas.testing.assert_geodataframe_equal."""

    def get_test_gdfs(self) -> TestGeoDataFrames:
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
        )

    def run(self) -> None:
        """Run integration test.

        Raises
        ------
            AssertionError: If test fails.

        """
        input_data, _reference_data, result, control = self.get_test_gdfs()

        assert input_data.crs == result.crs
        assert input_data.crs == control.crs

        # TODO: change these to failure states? However, that would
        # requires changes to some algorithms, so only warn for now.
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

        assert_geodataframe_equal(
            result,
            control,
            *self.assert_function_kwargs,
        )

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
