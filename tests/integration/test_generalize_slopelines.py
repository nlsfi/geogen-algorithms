#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from geogenalg.testing import AlgorithmTestInput
from tests.integration.runner import IntegrationTest


def test_generalize_slopelines(slopelines_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=slopelines_input,
        check_missing_reference=True,
        assert_function_arguments={
            "check_less_precise": False,
        },
    ).run()
