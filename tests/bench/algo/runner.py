#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from collections.abc import Callable
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

    def run(
        self,
        fixture: BenchmarkFixture,
        record_property: Callable | None = None,
    ) -> None:
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

        feature_count = alg_data.shape[0]
        mean_duration_s = fixture.stats.stats.mean

        if mean_duration_s > 0 and record_property is not None:
            features_per_s = feature_count / mean_duration_s
            record_property("features_per_sec", str(round(features_per_s, 2)))

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

            assert_geodataframe_equal(result, control)
