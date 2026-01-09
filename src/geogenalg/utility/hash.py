#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from secrets import token_hex
from typing import TypeVar, cast

from geopandas import GeoDataFrame
from pandas import DataFrame, Index

_Data = TypeVar("_Data", bound=DataFrame | GeoDataFrame)


def reset_with_random_hash_index(data: _Data) -> _Data:  # noqa: UP047, bump min python version to 3.12 or use 3.11 as ruff target
    """Reset the index with random SHA256-hash like hex string.

    Applies a new randomly generated index on the dataframe to indicate
    that the identities of the output are not preserved.

    Returns:
        Input dataframe with the index reset.

    """
    return cast(
        "_Data",
        data.set_index(
            Index(
                [token_hex(256 // 8) for _ in range(len(data.index))], dtype="string"
            ),
            inplace=False,
        ),
    )
