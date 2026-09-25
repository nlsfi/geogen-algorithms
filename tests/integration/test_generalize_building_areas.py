#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from geogenalg.testing import AlgorithmTestInput
from tests.integration.runner import ExpectedResultColumns, IntegrationTest


def test_generalize_building_areas(building_areas_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=building_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["building_function_id"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="none",
        ),
    ).run()
