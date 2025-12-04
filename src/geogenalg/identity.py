#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from hashlib import sha256

from geopandas import GeoDataFrame
from pandas.api.types import is_string_dtype


def hash_duplicate_indexes(
    data: GeoDataFrame,
    hash_prefix: str,
) -> GeoDataFrame:
    """Change any duplicate indexes to a hash value.

    It is required that the input has a string index.

    Hashing is done with SHA256 and its input is the concatenation of the hash
    prefix, the original index and the WKT of the part's geometry.

    Args:
    ----
        data: GeoDataFrame to be processed.
        hash_prefix: Prefix to use in hash function input.

    Returns:
    -------
        GeoDataFrame with duplicate indexes replaced with hashes.

    Raises:
    ------
        ValueError: If input GeoDataFrame does not have a string index.

    """
    if not is_string_dtype(data.index):
        msg = "GeoDataFrame must have a string index."
        raise ValueError(msg)

    gdf = data.copy()

    temp_index = "__temp_index"

    gdf[temp_index] = gdf.index

    mask = gdf.duplicated(keep=False, subset=[temp_index])
    gdf.loc[mask, temp_index] = (
        hash_prefix + gdf.loc[mask].index + gdf.loc[mask].geometry.to_wkt()
    )
    gdf.loc[mask, temp_index] = gdf.loc[mask, temp_index].apply(
        lambda hash_input: sha256(hash_input.encode()).hexdigest()
    )

    gdf = gdf.set_index(temp_index)
    gdf.index.name = data.index.name

    return gdf
