#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import ClassVar

from geopandas import GeoDataFrame
from pandas import Index
from pandas.testing import assert_index_equal
from shapely import Point

from geogenalg.application import BaseAlgorithm, supports_identity


def test_index_reset_without_identity_support():
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    input_data = GeoDataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "geom": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        index=["abc", "def", "ghi"],
        geometry="geom",
    )

    output_data = MockAlg().execute(input_data, {})

    assert len(output_data.index.intersection(["abc", "def", "ghi"])) == 0


def test_index_not_reset_with_identity_support():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    input_data = GeoDataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "geom": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        index=["abc", "def", "ghi"],
        geometry="geom",
    )

    output_data = MockAlg().execute(input_data, {})

    assert_index_equal(output_data.index, input_data.index)


def test_input_index_converted_to_strings_for_algorithm_use():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    input_data = GeoDataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "geom": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        index=[111, 222, 333],
        geometry="geom",
    )

    output_data = MockAlg().execute(input_data, {})

    assert_index_equal(output_data.index, Index(["111", "222", "333"], dtype="string"))


def test_geometry_column_does_not_change():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data.copy().rename_geometry("other_geom")

    input_data = GeoDataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "geom": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        index=[111, 222, 333],
        geometry="geom",
    )

    output_data = MockAlg().execute(input_data, {})

    assert output_data.geometry.name == "geom"
