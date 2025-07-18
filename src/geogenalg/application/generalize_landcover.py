#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from shapelysmooth import chaikin_smooth

from geogenalg import attributes, selection


@dataclass
class AlgorithmOptions:
    """Options for generalize land cover algorithm.

    Attributes:
        buffer_constant: Constant used for buffering polygons
        simplification_tolerance: Tolerance used for geometry simplification
        area_threshold: Minimum polygon area to retain
        hole_threshold: Minimum area of holes to retain
        smoothing: If True, polygons will be smoothed

    """

    buffer_constant: float
    simplification_tolerance: float
    area_threshold: float
    hole_threshold: float
    smoothing: bool


def create_generalized_landcover(
    input_path: Path | str,
    layer_name: str,
    options: AlgorithmOptions,
    output_path: str,
) -> None:
    """Create GeoDataFrame and pass it to the generalization function.

    Args:
    ----
        input_path: Path to the input GeoPackage
        layer_name: Name of the layer for land cover polygons
        options: Algorithm parameters for generalize land cover
        output_path: Path to save the output GeoPackage

    Raises:
    ------
        FileNotFoundError: If the input_path does not exist

    """
    if not Path(input_path).resolve().exists():
        raise FileNotFoundError

    landcover_gdf = gpd.read_file(input_path, layer=layer_name)

    result = generalize_landcover(landcover_gdf, options)

    result.to_file(output_path, driver="GPKG")


def generalize_landcover(
    landcover_gdf: gpd.GeoDataFrame,
    options: AlgorithmOptions,
) -> gpd.GeoDataFrame:
    """Generalize the polygon layer representing a land cover class.

    This algorithm can process only one polygon layer at a time, as generalization
    parameters need to be adjusted between different classes and scales to achieve
    a good cartographic result.

    Args:
    ----
        landcover_gdf: A GeoDataFrame containing the land cover polygons
        options: Algorithm parameters for generalize land cover

    Returns:
    -------
        Generalized land cover polygons

    """
    result_gdf = landcover_gdf.copy()

    # Create a buffer of size buffer_constant to close narrow gaps between polygons
    result_gdf.geometry = result_gdf.geometry.buffer(
        options.buffer_constant, cap_style="square", join_style="bevel"
    )
    result_gdf = result_gdf.dissolve(as_index=False)

    # Create a double negative buffer to remove narrow polygon parts
    result_gdf.geometry = result_gdf.geometry.buffer(
        -2 * options.buffer_constant, cap_style="square", join_style="bevel"
    )

    # Restore polygons to their original size with a positive buffer
    result_gdf.geometry = result_gdf.geometry.buffer(
        options.buffer_constant, cap_style="square", join_style="bevel"
    )

    result_gdf = result_gdf.dissolve(as_index=False)
    result_gdf = result_gdf.explode(index_parts=False)

    # Simplify the polygons
    result_gdf.geometry = result_gdf.geometry.simplify(
        options.simplification_tolerance, preserve_topology=True
    )

    # Smooth the polygons if smoothing is wanted
    if options.smoothing:
        geometries = list(result_gdf.geometry)
        smoothed_geometries = [chaikin_smooth(geom) for geom in geometries]
        result_gdf.geometry = smoothed_geometries

    # Remove polygons smaller than the area_threshold
    result_gdf = selection.remove_small_polygons(result_gdf, options.area_threshold)

    # Remove holes smaller than the hole_threshold and return
    result_gdf = selection.remove_small_holes(result_gdf, options.hole_threshold)

    return attributes.inherit_attributes(landcover_gdf, result_gdf)
