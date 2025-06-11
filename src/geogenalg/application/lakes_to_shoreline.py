from __future__ import annotations

from itertools import combinations

import geopandas as gpd


def extract_shoreline_from_generalized_lakes(
    original_shoreline: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Extracts the boundaries from the lakes and calculates their
    shoreline type from the original shoreline.

    Returns a GeoDataFrame of the generalized shoreline.
    """
    gdf = gpd.GeoDataFrame(
        {"lake_id": lakes.index}, geometry=lakes.boundary, crs=lakes.crs
    )

    buffered_shoreline = gpd.GeoDataFrame(
        original_shoreline["shoreline_type_id"],
        geometry=original_shoreline.buffer(10),
        crs=original_shoreline.crs,
    )

    # NOTE: The buffering uses a round cap style instead of a flat one because with a flat one
    # you get small gaps between the buffered sections leading to null values and extra vertices
    # in the proceeding identity operation, which are more complex to address. By buffering over
    # the end points you can make the buffered sections have overlapping sections which can then be
    # removed, ensuring topological correctness.

    # NOTE: As a downside, this method can cause the start and end points of a section with a
    # different category to its surroundings to shift by about the distance which is used for
    # the buffer operation. However the mismatch is hardly noticeable on the scale this algorithm
    # is intended for (1: 50 000) so with the additional complexity arising from the flat cap
    # style this seemed like the better option.

    buffered_geom = buffered_shoreline.geometry
    for point_1_idx, point_2_idx in combinations(buffered_geom.index, 2):
        if buffered_geom.loc[point_1_idx].intersects(buffered_geom.loc[point_2_idx]):
            buffered_geom.loc[point_2_idx] -= buffered_geom.loc[point_1_idx]

    gdf = (
        gpd.overlay(gdf, buffered_shoreline, how="identity")
        .dissolve(by=["lake_id", "shoreline_type_id"], as_index=False)
        .drop("lake_id", axis=1)
    )

    gdf["geometry"] = gdf.geometry.remove_repeated_points()
    gdf["geometry"] = gdf.geometry.line_merge()

    return gdf.explode()
