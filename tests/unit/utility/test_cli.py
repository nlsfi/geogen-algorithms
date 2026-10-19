#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import ClassVar, override

import pytest
from geopandas import GeoDataFrame
from pydantic import Field

from geogenalg.application import BaseAlgorithm
from geogenalg.utility.cli import (
    NamedGeoPackageURI,
    dump_cli_parameters,
    get_basealgorithm_attribute_docstrings,
    get_class_attribute_docstrings,
    try_parse_int,
)


def test_get_class_attribute_docstrings():
    class TestClass:
        attribute_1: str = "something"
        """Attribute 1 docstring"""
        attribute_2: int = 10
        """Attribute 2 docstring"""
        attribute_3: int = 10  # doesn't have a docstring

    assert get_class_attribute_docstrings(TestClass) == {
        "attribute_1": "Attribute 1 docstring",
        "attribute_2": "Attribute 2 docstring",
    }


def test_get_basealgorithm_docstrings():
    class MockAlgParent(BaseAlgorithm):
        parent_attribute: str = "something"
        """Parent attribute."""

        def _execute(
            self,
            data: GeoDataFrame,  # noqa: ARG002
            reference_data: dict[str, GeoDataFrame],  # noqa: ARG002
        ) -> GeoDataFrame:
            return GeoDataFrame()

    class MockAlgSubclass(MockAlgParent):
        subclass_attribute: str = "something"
        """Subclass attribute."""

        def _execute(
            self,
            data: GeoDataFrame,  # noqa: ARG002
            reference_data: dict[str, GeoDataFrame],  # noqa: ARG002
        ) -> GeoDataFrame:
            return GeoDataFrame()

    assert get_basealgorithm_attribute_docstrings(MockAlgSubclass) == {
        "parent_attribute": "Parent attribute.",
        "subclass_attribute": "Subclass attribute.",
        "repair_result_geometries": "Specify if repairing invalid and non-simple geometries should be attempted.",
    }


@pytest.mark.parametrize(
    ("input_string", "expected"),
    [
        (
            "100",
            100,
        ),
        (
            "str:100",
            "100",
        ),
        (
            "str:100:",
            "str:100:",
        ),
        (
            "str:100.0",
            "str:100.0",
        ),
        (
            "100.0",
            "100.0",
        ),
        (
            "test_string",
            "test_string",
        ),
        (
            "-10",
            -10,
        ),
    ],
    ids=[
        "str_to_int",
        "special_encoding",
        "special_encoding_but_not_integer",
        "special_encoding_float",
        "float",
        "stays_str",
        "negative",
    ],
)
def test_try_parse_int(input_string: str, expected: NamedGeoPackageURI):
    assert try_parse_int(input_string) == expected


def test_dump_cli_parameters():
    class MockAlg(BaseAlgorithm):
        class_var: ClassVar[str] = "should_be_ignored"

        string_parameter: str = Field("default_str")
        int_parameter: int = Field(2, ge=0)
        frozenset_parameter: frozenset[str | int] = frozenset()
        do_something: bool = True
        optional_parameter: str | None = None

        @override
        def _execute(
            self,
            data: GeoDataFrame,
            reference_data: dict[str, GeoDataFrame],
        ) -> GeoDataFrame:
            return data

    all_defaults = MockAlg()
    assert dump_cli_parameters(all_defaults, exclude_defaults=True) == ""

    # 2. Modified instance with exclude_defaults=True
    some_defaults = MockAlg(
        int_parameter=10,
        frozenset_parameter=frozenset(["something", "else"]),
        do_something=False,
    )
    some_defaults_result = dump_cli_parameters(some_defaults, exclude_defaults=True)

    assert "--no-do-something" in some_defaults_result
    assert "--do-something" not in some_defaults_result

    assert "--int-parameter=10" in some_defaults_result
    assert "--string-parameter" not in some_defaults_result
    assert (
        "--frozenset-parameter=else --frozenset-parameter=something"
        in some_defaults_result
    )

    assert "--optional-parameter" not in some_defaults_result

    full_result = dump_cli_parameters(all_defaults, exclude_defaults=False)

    assert "--string-parameter=default_str" in full_result
    assert "--int-parameter=2" in full_result
    assert "--do-something" in full_result
    assert "class-var" not in full_result
    assert "should-be-ignored" not in full_result
    assert "frozenset-parameter" not in full_result  # empty by default
    assert "optional-parameter" not in full_result

    # Handle whitespaces
    whitespaces = MockAlg(string_parameter="hello world")
    whitespaces_result = dump_cli_parameters(whitespaces, exclude_defaults=True)
    assert (
        "--string-parameter='hello world'" in whitespaces_result
        or '--string-parameter="hello world"' in whitespaces_result
    )
