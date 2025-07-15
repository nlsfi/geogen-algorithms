#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from cartagen.utils.partitioning.network import network_faces

from geogenalg import continuity, merge, selection


@dataclass
class AlgorithmOptions:
    """Options for generalize fences algorithm.

    Attributes:
        closing_fence_area_threshold: Minimum area for a fence-enclosed region
        closing_fence_area_with_mast_threshold: Minimum area for a fence-enclosed
              region containing a mast
        fence_length_threshold: Minimum length for a fence line
        fence_length_threshold_in_closed_area: Minimum length for a fence line within a
              closed area
        simplification_tolerance: Tolerance used for geometry simplification
        gap_threshold: Maximum gap between two fence lines to be connected with
              a helper line
        attribute_for_line_merge: Name of the attribute to determine which line
              features can be merged

    """

    closing_fence_area_threshold: float
    closing_fence_area_with_mast_threshold: float
    fence_length_threshold: float
    fence_length_threshold_in_closed_area: float
    simplification_tolerance: float
    gap_threshold: float
    attribute_for_line_merge: str


def create_generalized_fences(
    input_path: Path | str,
    fences_layer_name: str,
    masts_layer_name: str,
    options: AlgorithmOptions,
    output_path: str,
) -> None:
    """Create GeoDataFrames and pass them to the generalization function.

    Args:
    ----
        input_path: Path to the input GeoPackage
        fences_layer_name: Name of the layer for fences
        masts_layer_name: Name of the layer for masts
        options: Algorithm parameters for generalize fences
        output_path: Path to save the output GeoPackage

    Raises:
    ------
        FileNotFoundError: If the input_path does not exist

    """
    if not Path(input_path).resolve().exists():
        raise FileNotFoundError

    fences_gdf = gpd.read_file(input_path, layer=fences_layer_name)

    masts_gdf = gpd.read_file(input_path, layer=masts_layer_name)

    result = generalize_fences(
        fences_gdf,
        masts_gdf,
        options,
    )

    result.to_file(output_path, driver="GPKG")


def generalize_fences(
    fences_gdf: gpd.GeoDataFrame, masts_gdf: gpd.GeoDataFrame, options: AlgorithmOptions
) -> gpd.GeoDataFrame:
    """Generalize the LineString layer representing fences.

    Args:
    ----
        fences_gdf: A GeoDataFrame containing the fence lines to be generalized
        masts_gdf: A GeoDataFrame containing masts
        options: Algorithm parameters for generalize fences

    Returns:
    -------
        Generalized fence lines

    """
    result_gdf = fences_gdf.copy()

    # Merge connecting lines with the same attribute value
    result_gdf = merge.merge_connecting_lines_by_attribute(
        result_gdf, options.attribute_for_line_merge
    )

    # Generate helper lines to close small gaps
    helper_lines_gdf = continuity.connect_nearby_endpoints(
        result_gdf, options.gap_threshold
    )

    # Combine original fence lines with helper lines
    combined_gdf: gpd.GeoDataFrame
    combined_gdf = pd.concat([result_gdf, helper_lines_gdf], ignore_index=True)

    # Calculate the CartaGen network faces to fill closing line geometries
    faces = network_faces(list(combined_gdf.geometry), convex_hull=False)
    faces_gdf = gpd.GeoDataFrame(geometry=list(faces))

    # Dissolve adjacent polygons into larger contiguous areas
    faces_gdf = faces_gdf.dissolve(as_index=False)
    faces_gdf = faces_gdf.union_all()

    # Select fence lines shorter than the threshold for enclosed areas
    short_lines = result_gdf[
        result_gdf.geometry.length < options.fence_length_threshold_in_closed_area
    ]

    # Remove short fence lines that are within closed polygonal areas
    to_remove_idx = short_lines[short_lines.geometry.apply(faces_gdf.contains)].index
    result_gdf = result_gdf.drop(index=to_remove_idx)

    # Convert MultiPolygon result back into a proper GeoDataFrame
    polygons = list(faces_gdf.geoms)
    faces_gdf = gpd.GeoDataFrame(geometry=polygons, crs=fences_gdf.crs)

    # Remove polygons whose area exceeds the closing_fence_area_with_mast_threshold and
    # the closing_fence_area_threshold
    polygon_gdf_with_point, polygon_gdf_without_point = (
        selection.split_polygons_by_point_intersection(faces_gdf, masts_gdf)
    )
    polygon_gdf_with_point = selection.remove_large_polygons(
        polygon_gdf_with_point, options.closing_fence_area_with_mast_threshold
    )
    polygon_gdf_without_point = selection.remove_large_polygons(
        polygon_gdf_without_point, options.closing_fence_area_threshold
    )

    # Combine filtered polygons
    faces_gdf = pd.concat(
        [polygon_gdf_with_point, polygon_gdf_without_point], ignore_index=True
    )

    # Remove the surrounding fence lines of small closed areas with masts considered
    result_gdf = selection.remove_parts_of_lines_on_polygon_edges(result_gdf, faces_gdf)

    # Remove short fence lines
    result_gdf = selection.remove_disconnected_short_lines(
        result_gdf, options.fence_length_threshold
    )

    # Return simplified fence lines
    result_gdf.geometry = result_gdf.geometry.simplify(options.simplification_tolerance)

    return result_gdf
