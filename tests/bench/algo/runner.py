#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from geopandas.testing import assert_geodataframe_equal
from pytest_benchmark.fixture import BenchmarkFixture

from geogenalg.application import BaseAlgorithm
from geogenalg.testing import TestInputData, tile_test_data
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

TILE_TO_FEATURES_ENV_VAR = "GEOGENALG_BENCH_TILE_TO_FEATURES"


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

        if control is None:
            msg = "No control data found."
            raise ValueError(msg)

        tile_count = 1
        if (target := os.environ.get(TILE_TO_FEATURES_ENV_VAR)) is not None:
            alg_data, reference_data, tile_count = tile_test_data(
                alg_data,
                reference_data,
                int(target),
            )
            if record_property is not None:
                record_property("tiles", str(tile_count))

        result = fixture.pedantic(
            self.algorithm.execute,
            args=(alg_data, reference_data),
            rounds=3,
            iterations=1,
            warmup_rounds=1,
        )

        feature_count = alg_data.shape[0]

        if fixture.stats is not None:
            mean_duration_s = fixture.stats.stats.mean

            if mean_duration_s > 0 and record_property is not None:
                features_per_s = feature_count / mean_duration_s
                record_property("features_per_sec", str(round(features_per_s, 2)))
                record_property("feature_count", str(feature_count))
                record_property("mean_duration_s", str(round(mean_duration_s, 3)))

        # With tiling we can't really assert against the control so return early
        if tile_count > 1:
            return

        # Save result to temp file and read again as a GDF so that column
        # dtypes etc. match with the control
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.gpkg"
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

            assert_geodataframe_equal(
                result,
                control,
                check_like=True,
                check_less_precise=True,
            )
