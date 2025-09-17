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
from cartagen.algorithms import buildings
from shapely.geometry import Polygon

from geogenalg import analyze, exaggeration, merge, selection
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility import dataframe_processing, fix_geometries


@dataclass
class AlgorithmOptions:
    """Options for the building generalization algorithm.

    Attributes:
        area_threshold_for_all_buildings: Minimum area threshold applied to all
              buildings
        area_threshold_for_low_priority_buildings: Minimum area threshold for
              low-priority building classes
        side_threshold: Longest side length used to distinguish small vs. large
              buildings
        point_size: Side length of the square used to represent point buildings
        minimum_distance_to_isolated_building: Minimum distance to the nearest neighbor
              to retain an isolated building
        hole_threshold: Minimum area for holes to retain inside polygon buildings
        classes_for_low_priority_buildings: Building classes treated as low-priority
        classes_for_point_buildings: Building classes always represented as points
        classes_for_always_kept_buildings: Building classes that are always retained,
              regardless of thresholds
        unique_key_column: Column name containing the unique identifier
        building_class_column: Column name containing the building class information
        dissolve_members_column: Name of the column that lists the original polygons
              that have been dissolved
        original_area_column: Column name for storing the building's original area
        main_angle_column: Column name for storing the main angle of the building

    """

    area_threshold_for_all_buildings: float
    area_threshold_for_low_priority_buildings: float
    side_threshold: float
    point_size: float
    minimum_distance_to_isolated_building: float
    hole_threshold: float
    classes_for_low_priority_buildings: list[str] | list[int]
    classes_for_point_buildings: list[str] | list[int]
    classes_for_always_kept_buildings: list[str] | list[int]
    unique_key_column: str
    building_class_column: str
    dissolve_members_column: str = "dissolve_members"
    original_area_column: str = "original_area"
    main_angle_column: str = "main_angle"


def create_generalized_buildings(
    input_path: Path | str,
    options: AlgorithmOptions,
    output_path: str,
) -> None:
    """Create GeoDataFrames and pass them to the generalization function.

    Args:
    ----
        input_path: Path to the input GeoPackage
        options: Algorithm parameters for generalize buildings
        output_path: Path to save the output GeoPackage

    Raises:
    ------
        FileNotFoundError: If the input_path does not exist

    """
    if not Path(input_path).resolve().exists():
        raise FileNotFoundError

    layers_info = gpd.list_layers(input_path)
    layer_names = layers_info["name"].tolist()

    input_gdfs = [gpd.read_file(input_path, layer=name) for name in layer_names]

    result_gdfs = generalize_buildings(
        input_gdfs,
        options,
    )

    for layer_name, result_gdf in result_gdfs.items():
        result_gdf.to_file(output_path, layer=layer_name, driver="GPKG")


def generalize_buildings(
    building_gdfs: list[gpd.GeoDataFrame],
    options: AlgorithmOptions,
) -> dict[str, gpd.GeoDataFrame]:
    """Generalize buildings.

    Args:
    ----
        building_gdfs: List of GeoDataFrames containing buildings as polygons
              and/or points
        options: Algorithm parameters for generalize buildings

    Returns:
    -------
        A dictionary with:
            - "small_buildings": generalized point buildings
            - "large_buildings": generalized polygon buildings

    """
    polygon_buildings_gdf = dataframe_processing.group_gdfs_by_geometry_type(
        building_gdfs
    )["polygon_gdf"]
    point_buildings_gdf = dataframe_processing.group_gdfs_by_geometry_type(
        building_gdfs
    )["point_gdf"]

    polygon_buildings_gdf = _add_attributes_for_area_and_angle(
        polygon_buildings_gdf, options.original_area_column, options.main_angle_column
    )
    point_buildings_gdf = _add_attributes_for_area_and_angle(
        point_buildings_gdf, options.original_area_column, options.main_angle_column
    )

    polygon_buildings_gdf = _filter_buildings_by_area_and_class(
        polygon_buildings_gdf, options
    )
    point_buildings_gdf = _filter_buildings_by_area_and_class(
        point_buildings_gdf, options
    )

    polygon_buildings_gdf = _dissolve_touching_buildings(
        polygon_buildings_gdf, options.building_class_column, options.unique_key_column
    )

    always_point_buildings = polygon_buildings_gdf[options.building_class_column].isin(
        options.classes_for_point_buildings
    )

    # Classify only those buildings that are not flagged as "always_point_buildings"
    classified_gdf = analyze.classify_polygons_by_size_of_minimum_bounding_rectangle(
        polygon_buildings_gdf[~always_point_buildings],
        options.side_threshold,
    )

    # Create centroids for small and large polygons
    classified_gdf["small_polygons"] = pd.concat(
        [
            classified_gdf["small_polygons"],
            polygon_buildings_gdf[always_point_buildings],
        ]
    )
    small_building_centroids_gdf = classified_gdf["small_polygons"].copy()
    small_building_centroids_gdf.geometry = small_building_centroids_gdf.centroid

    large_building_centroids_gdf = classified_gdf["large_polygons"].copy()
    large_building_centroids_gdf.geometry = large_building_centroids_gdf.centroid

    # Combine original point buildings with centroids of small polygons
    small_buildings_gdf = pd.concat(
        [point_buildings_gdf, small_building_centroids_gdf], ignore_index=True
    )

    # Combine centroids of large polygons with all small building points
    all_building_centroids_gdf = gpd.GeoDataFrame(
        pd.concat(
            [large_building_centroids_gdf, small_buildings_gdf], ignore_index=True
        ),
        crs=point_buildings_gdf.crs,
    )

    small_buildings_gdf = generalize_point_buildings(
        small_buildings_gdf, all_building_centroids_gdf, options
    )

    large_buildings_gdf = generalize_polygon_buildings(
        classified_gdf["large_polygons"], options
    )

    return {
        "small_buildings": small_buildings_gdf,
        "large_buildings": large_buildings_gdf,
    }


def generalize_point_buildings(
    point_buildings_gdf: gpd.GeoDataFrame,
    all_buildings_gdf: gpd.GeoDataFrame,
    options: AlgorithmOptions,
) -> gpd.GeoDataFrame:
    """Generalize point buildings by reducing density based on proximity and priority.

    Args:
    ----
        point_buildings_gdf: A GeoDataFrame containing the centroids of small buildings
        all_buildings_gdf: A GeoDataFrame containing the centroids of all buildings
        options: Algorithm parameters for generalize buildings

    Returns:
    -------
        A GeoDataFrame containing generalized point buildings.

    """
    # Split buildings into low-priority and others (excluding always-kept)
    low_priority_buildings_gdf = point_buildings_gdf[
        point_buildings_gdf[options.building_class_column].isin(
            options.classes_for_low_priority_buildings
        )
        & ~point_buildings_gdf[options.building_class_column].isin(
            options.classes_for_always_kept_buildings
        )
    ]

    other_buildings_gdf = point_buildings_gdf[
        ~point_buildings_gdf[options.building_class_column].isin(
            options.classes_for_low_priority_buildings
        )
        & ~point_buildings_gdf[options.building_class_column].isin(
            options.classes_for_always_kept_buildings
        )
    ]

    # Reduce density separately for each group
    low_priority_buildings_gdf = selection.reduce_nearby_points_by_selecting(
        low_priority_buildings_gdf,
        all_buildings_gdf,
        options.minimum_distance_to_isolated_building,
        options.original_area_column,
        options.unique_key_column,
    )

    other_buildings_gdf = selection.reduce_nearby_points_by_selecting(
        other_buildings_gdf,
        all_buildings_gdf,
        options.point_size * 1.66,
        options.original_area_column,
        options.unique_key_column,
    )

    return pd.concat(
        [low_priority_buildings_gdf, other_buildings_gdf],
        ignore_index=True,
    )


def generalize_polygon_buildings(
    input_gdf: gpd.GeoDataFrame, options: AlgorithmOptions
) -> gpd.GeoDataFrame:
    """Generalize polygon buildings by simplifying and exaggerating.

    Args:
    ----
        input_gdf: A GeoDataFrame containing the polygon buildings
        options: Algorithm parameters for generalize buildings

    Returns:
    -------
        A GeoDataFrame containing generalized polygon buildings.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than
              Polygon or MultiPolygon geometries.

    """
    if not all(input_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])):
        msg = (
            "generalize_polygon_buildings only supports "
            + "Polygon or MultiPolygon geometries."
        )
        raise GeometryTypeError(msg)

    result_gdf = input_gdf.copy()

    # Simplify polygons using CartaGen Ruas simplification
    result_gdf = _simplify_buildings(result_gdf, 5)
    simplified_gdf = result_gdf.copy()

    # Buffer the narrow building parts by 3 meters
    narrow_parts_gdf = exaggeration.extract_narrow_polygon_parts(
        result_gdf, options.point_size
    )
    narrow_parts_gdf.geometry = narrow_parts_gdf.geometry.buffer(
        (3), cap_style="flat", join_style="mitre"
    )

    # Dissolve the expanded narrow parts and already large enough parts
    result_gdf = pd.concat([simplified_gdf, narrow_parts_gdf], ignore_index=True)
    result_gdf = merge.dissolve_and_inherit_attributes(
        result_gdf, options.unique_key_column, options.unique_key_column
    )

    # Simplify polygons again using CartaGen Ruas simplification
    result_gdf = _simplify_buildings(result_gdf, 7.5)

    # Subtract buildings from the background
    result_gdf = fix_geometries.drop_empty_geometries(result_gdf)
    result_gdf = fix_geometries.fix_invalid_geometries(result_gdf)
    simplified_gdf = fix_geometries.drop_empty_geometries(simplified_gdf)
    simplified_gdf = fix_geometries.fix_invalid_geometries(simplified_gdf)
    bounding_polygon = simplified_gdf.union_all().envelope
    difference_gdf = gpd.GeoDataFrame(
        geometry=[bounding_polygon.difference(result_gdf.union_all())],
        crs=input_gdf.crs,
    )

    # Find areas where buildings are less than 12 meters apart
    narrow_gaps_gdf = exaggeration.extract_narrow_polygon_parts(difference_gdf, 12)

    # Remove areas from narrow gaps where actual buildings exist —
    # do not want to widen gaps that already contain building
    narrow_gaps_gdf.geometry = narrow_gaps_gdf.geometry.difference(
        input_gdf.union_all()
    )

    # Expand narrow gaps
    narrow_gaps_gdf = narrow_gaps_gdf.geometry.buffer(
        (3.5), cap_style="flat", join_style="mitre"
    )

    # Subtract expanded gaps from the generalized buildings
    result_gdf.geometry = result_gdf.geometry.difference(narrow_gaps_gdf.union_all())

    # Dissolve buildings with the same building class
    result_gdf = merge.dissolve_and_inherit_attributes(
        result_gdf, options.building_class_column, options.unique_key_column
    )

    # Simplify polygons once again
    result_gdf = _simplify_buildings(result_gdf, 8)

    # Remove holes smaller than the hole_threshold size
    result_gdf = selection.remove_small_holes(result_gdf, options.hole_threshold)

    # Return buildings larger than a square with side length equal to point_size
    return selection.remove_small_polygons(result_gdf, options.point_size**2)


def _dissolve_touching_buildings(
    input_gdf: gpd.GeoDataFrame, building_class_column: str, unique_key_column: str
) -> gpd.GeoDataFrame:
    """Slightly buffer buildings and dissolve adjacent buildings of the same class.

    Returns
    -------
        A GeoDataFrame containing dissolved polygon buildings.

    """
    input_gdf.geometry = input_gdf.buffer(0.1, cap_style="flat", join_style="mitre")
    input_gdf = merge.dissolve_and_inherit_attributes(
        input_gdf,
        building_class_column,
        unique_key_column,
    )
    input_gdf.geometry = input_gdf.buffer(-0.1, cap_style="flat", join_style="mitre")

    return input_gdf


def _filter_buildings_by_area_and_class(
    input_gdf: gpd.GeoDataFrame, options: AlgorithmOptions
) -> gpd.GeoDataFrame:
    """Filter buildings based on class-based area thresholds.

    Returns
    -------
        A GeoDataFrame with buildings that are too small filtered out.

    """
    if input_gdf.empty:
        return input_gdf.copy()

    return input_gdf[
        (
            (
                input_gdf[options.original_area_column]
                >= options.area_threshold_for_all_buildings
            )
            & ~(
                (
                    input_gdf[options.building_class_column].isin(
                        options.classes_for_low_priority_buildings
                    )
                )
                & (
                    input_gdf[options.original_area_column]
                    < options.area_threshold_for_low_priority_buildings
                )
            )
        )
        | input_gdf[options.building_class_column].isin(
            options.classes_for_always_kept_buildings
        )
    ]


def _add_attributes_for_area_and_angle(
    input_gdf: gpd.GeoDataFrame,
    original_area_column: str = "original_area",
    main_angle_column: str = "main_angle",
) -> gpd.GeoDataFrame:
    """Ensure every building geometry has attributes for original area and angle.

    Returns
    -------
        A GeoDataFrame with columns for original area and main angle.

    """
    if input_gdf.empty:
        if original_area_column not in input_gdf.columns:
            input_gdf[original_area_column] = pd.Series(dtype=float)
        if main_angle_column not in input_gdf.columns:
            input_gdf[main_angle_column] = pd.Series(dtype=float)
        return input_gdf

    geom_types = input_gdf.geometry.geom_type.unique()
    if all(geom_type in {"Polygon", "MultiPolygon"} for geom_type in geom_types):
        if original_area_column not in input_gdf.columns:
            input_gdf[original_area_column] = input_gdf.geometry.area
        if main_angle_column not in input_gdf.columns:
            input_gdf[main_angle_column] = input_gdf.geometry.apply(
                analyze.calculate_main_angle
            )

    # If the area and angle attributes of a point building have been lost, they are
    # replaced with 0.0
    elif all(geom_type in {"Point", "MultiPoint"} for geom_type in geom_types):
        if original_area_column not in input_gdf.columns:
            input_gdf[original_area_column] = 0.0
        if main_angle_column not in input_gdf.columns:
            input_gdf[main_angle_column] = 0.0
            input_gdf[main_angle_column] = pd.Series(dtype=float)

    return input_gdf


def _simplify_buildings(
    input_gdf: gpd.GeoDataFrame, edge_threshold: float
) -> gpd.GeoDataFrame:
    """Simplifiy building geometries using the Cartagen Ruas algorithm.

    Returns
    -------
        A GeoDataFrame containing simplified polygon geometries.

    Raises
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than
              Polygon or MultiPolygon geometries.

    """
    if not all(input_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])):
        msg = "Simplify buildings only supports Polygon or MultiPolygon geometries."
        raise GeometryTypeError(msg)

    result_gdf = input_gdf.copy().explode(index_parts=False).reset_index(drop=True)

    def _simplify_building(geometry: Polygon) -> list:
        return buildings.simplify_building(geometry, edge_threshold)

    result_gdf.geometry = result_gdf.geometry.apply(_simplify_building)
    return result_gdf
