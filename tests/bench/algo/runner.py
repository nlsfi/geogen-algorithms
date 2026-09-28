#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from dataclasses import dataclass
from tempfile import NamedTemporaryFile

from geopandas.testing import assert_geodataframe_equal
from pytest_benchmark.fixture import BenchmarkFixture

from geogenalg.application import BaseAlgorithm
from geogenalg.testing import TestInputData
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index


@dataclass(frozen=True, kw_only=True)
class AlgorithmBenchmark:
    """Class for defining an benchmark for a specific algorithm."""

    input_data: TestInputData
    algorithm: BaseAlgorithm

    def run(self, fixture: BenchmarkFixture) -> None:
        if self.input_data.unique_id_column is None:
            msg = "Unique id column must be set."
            raise ValueError(msg)

        alg_data, control, reference_data = self.input_data.read()

        result = fixture.pedantic(
            self.algorithm.execute,
            args=(alg_data, reference_data),
            rounds=3,
            iterations=1,
            warmup_rounds=1,
        )

        # Save result to temp file and read again as a GDF so that column
        # dtypes etc. match with the control
        with NamedTemporaryFile(suffix=".gpkg", delete=True) as temp_file:
            output_path = temp_file.name
            result.to_file(
                output_path,
                layer="result",
                driver="GPKG",
            )

            result = read_gdf_from_file_and_set_index(
                output_path,
                self.input_data.unique_id_column,
                layer="result",
            )

            assert_geodataframe_equal(control, result)
