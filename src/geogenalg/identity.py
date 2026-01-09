#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from hashlib import sha256

from geopandas import GeoDataFrame
from pandas import DataFrame, Index
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


def hash_index_from_old_ids(
    data: DataFrame, hash_prefix: str, old_ids_column: str, *, drop_old_ids: bool = True
) -> DataFrame:
    """Change dataframe index to a hash value based on an old IDs column.

    It is required that the input has a string index.

    The old IDs may be a result of a merging rows together, and must contain a
    tuple or a list, or a null value.

    The hashing will use SHA256 and the input will be the given hash prefix and
    all the old IDs concatenated together.

    If a row has an empty or null value for old IDs, the original index is retained.

    Args:
    ----
        data: (Geo)DataFrame to process.
        hash_prefix: Prefix to use in hash input, can be used to identify an algorithm
            so that two algorithms can't produce the same output.
        old_ids_column: Name of column where old IDs are found.
        drop_old_ids: Whether old_ids_column should be dropped after updating the index.

    Returns:
    -------
        DataFrame with the processed index.

    Raises:
    ------
        ValueError: If input data does not have a string index.
        TypeError: If a non-null old IDs value is not a tuple or a list.

    """
    if not is_string_dtype(data.index):
        msg = "DataFrame must have a string index."
        raise ValueError(msg)

    gdf = data.copy()

    new_index = []
    for index, row in gdf.iterrows():
        old_ids = row[old_ids_column]

        if not old_ids:
            new_index.append(index)
            continue

        if not isinstance(old_ids, tuple | list):
            msg = "Column with old ids should contain a list or a tuple."
            raise TypeError(msg)

        if len(old_ids) == 1:
            new_index.append(index)
            continue

        new_index.append(
            sha256(
                hash_prefix.encode() + "".join(sorted(old_ids)).encode(),
            ).hexdigest()
        )

    gdf.index = Index(new_index)
    gdf.index.name = data.index.name

    if drop_old_ids:
        gdf = gdf.drop(old_ids_column, axis=1)

    return gdf
