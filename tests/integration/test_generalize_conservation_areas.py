#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from geogenalg.testing import AlgorithmTestInput
from tests.integration.runner import IntegrationTest


def test_generalize_conservation_areas(
    conservation_areas_input: AlgorithmTestInput,
) -> None:
    IntegrationTest(
        algorithm_input=conservation_areas_input,
        check_missing_reference=False,
        dummy_data_mandatory_columns=frozenset(["layer"]),
    ).run()
