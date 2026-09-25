#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from geogenalg.testing import AlgorithmTestInput
from tests.integration.runner import IntegrationTest


def test_generalize_shared_paths(shared_paths_input: AlgorithmTestInput) -> None:
    IntegrationTest(
        algorithm_input=shared_paths_input,
        check_missing_reference=True,
    ).run()
