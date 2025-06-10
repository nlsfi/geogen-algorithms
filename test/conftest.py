#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest


def _testdata_path() -> Path:
    return Path(__file__).resolve().parent / "testdata"


@pytest.fixture
def boulders_in_water_10k_sourcedata_path() -> Path:
    return _testdata_path() / "boulders_in_water_10k.gpkg"


@pytest.fixture
def boulders_in_water_no_land_sourcedata_path() -> Path:
    return _testdata_path() / "boulders_in_water_no_land.gpkg"
