#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

import pytest


@pytest.fixture
def testdata_path() -> Path:
    return Path(__file__).resolve().parent / "testdata"
