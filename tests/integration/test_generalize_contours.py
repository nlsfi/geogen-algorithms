#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from geogenalg.testing import AlgorithmTestInput
from tests.integration.runner import IntegrationTest


def test_generalize_contours(contours_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=contours_input,
        check_missing_reference=False,
        assert_function_arguments={
            "check_less_precise": True,
        },
        dummy_data_mandatory_columns=frozenset(["n60_elevation_value"]),
    ).run()
