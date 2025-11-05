#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path
from tempfile import TemporaryDirectory

from geopandas import read_file
from pandas.testing import assert_frame_equal

from geogenalg.application.generalize_shoreline import GeneralizeShoreline


def test_extract_shoreline_from_generalized_lakes(
    testdata_path: Path,
) -> None:
    temp_dir = TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    test_file = testdata_path / "lakes_to_shoreline.gpkg"

    original_shoreline = read_file(
        test_file,
        layer="original_shoreline",
    )
    generalized_lakes = read_file(
        test_file,
        layer="generalized_lakes",
    )
    control = read_file(
        test_file,
        layer="control",
    )

    GeneralizeShoreline().execute(
        original_shoreline, {"areas": generalized_lakes}
    ).to_file(output_path, layer="result")

    result = read_file(output_path, layer="result")

    assert_frame_equal(result, control)
