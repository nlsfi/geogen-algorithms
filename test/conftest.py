#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest


@pytest.fixture
def testdata_path() -> Path:
    return Path(__file__).resolve().parent / "testdata"
