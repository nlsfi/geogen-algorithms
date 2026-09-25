#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from geogenalg.testing import AlgorithmTestInput
from tests.integration.runner import ExpectedResultColumns, IntegrationTest


def test_generalize_watercourse_areas(watercourse_areas_input: AlgorithmTestInput):
    IntegrationTest(
        algorithm_input=watercourse_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["shoreline_type_id"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
            mandatory_extra_columns=frozenset(["feature_type"]),
        ),
    ).run()
