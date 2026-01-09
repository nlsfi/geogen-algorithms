#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from geopandas import GeoDataFrame
from pandas.api.types import is_string_dtype

from geogenalg.identity import hash_duplicate_indexes


def explode_and_hash_id(
    data: GeoDataFrame,
    hash_prefix: str,
) -> GeoDataFrame:
    """Explode any multigeometries and set their index as a hash value.

    It is required that the input has a string index.

    If a feature is of a multigeometry type but has only single part, it
    will be changed to a single geometry and the original ID retained.

    Hashing is done with SHA256 and its input is the concatenation of the hash
    prefix, the original index and the WKT of the part's geometry.

    Args:
    ----
        data: GeoDataFrame to be processed.
        hash_prefix: Prefix to use in hash function input.

    Returns:
    -------
        GeoDataFrame with unchanged (if any) and exploded (if any) features.

    Raises:
    ------
        ValueError: If input GeoDataFrame does not have a string index.

    """
    if not is_string_dtype(data.index):
        msg = "GeoDataFrame must have a string index."
        raise ValueError(msg)

    length_before = len(data.index)
    gdf = data.explode()

    if len(gdf.index) == length_before:
        # Nothing exploded, no need to keep going.
        return gdf

    return hash_duplicate_indexes(gdf, hash_prefix)
