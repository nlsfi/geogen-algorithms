#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from cartagen.algorithms import buildings
from geopandas import GeoDataFrame
from pandas import Series, concat

from geogenalg.analyze import (
    calculate_main_angle,
    classify_polygons_by_size_of_minimum_bounding_rectangle,
)
from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.exaggeration import extract_narrow_polygon_parts
from geogenalg.identity import hash_index_from_old_ids
from geogenalg.merge import dissolve_and_inherit_attributes
from geogenalg.selection import (
    reduce_nearby_points_by_selecting,
    remove_small_holes,
    remove_small_polygons,
)
from geogenalg.utility.fix_geometries import (
    drop_empty_geometries,
    fix_invalid_geometries,
)
from geogenalg.utility.validation import check_gdf_geometry_type

SIMPLIFY_EDGE_THRESHOLD = 5
SIMPLIFY_EDGE_THRESHOLD_AFTER_NARROW_PARTS = 7.5
SIMPLIFY_EDGE_THRESHOLD_AFTER_NARROW_GAPS = 8
NARROW_GAPS_THRESHOLD = 12
BUFFER_SIZE_FOR_NARROW_PARTS = 3
BUFFER_SIZE_FOR_NARROW_GAPS = 3.5


@supports_identity
@dataclass(frozen=True)
class GeneralizeBuildings(BaseAlgorithm):
    """Generalizes buildings.

    No reference data used.

    Input data should contain Polygons.

    - Removes buildings based on area and building type.
    - Buildings are classified into small and large buildings
    - Small buildings are turned to points
    - Large buildings remain as polygons
    - Point buildings are thinned
    - Polygon buildings are simplified
    - Narrow gaps and part in polygon buildings are exaggerated
    - Small holes in polygon buildings are removed
    """

    area_threshold_for_all_buildings: float = 5
    """Minimum area threshold applied to all buildings."""
    area_threshold_for_low_priority_buildings: float = 100
    """Minimum area threshold for low-priority building classes."""
    side_threshold: float = 30
    """Longest side length used to distinguish small vs. large buildings."""
    point_size: float = 15
    """Side length of the square used to represent point buildings."""
    minimum_distance_to_isolated_building: float = 200
    """Minimum distance to the nearest neighbor to retain an isolated
    building."""
    hole_threshold: float = 75
    """Minimum area for holes to retain inside polygon buildings."""
    classes_for_low_priority_buildings: list[int | str] | None = None
    """Building classes treated as low-priority."""
    classes_for_point_buildings: list[int | str] | None = None
    """Building classes always represented as points."""
    classes_for_always_kept_buildings: list[int | str] | None = None
    """Building classes that are always retained, regardless of thresholds."""
    unique_key_column: str = "mtk_id"
    """Column name containing the unique identifier."""
    building_class_column: str = "kayttotarkoitus"
    """Column name containing the building class information."""
    original_area_column: str = "original_area"
    """Column name for storing the building's original area."""
    main_angle_column: str = "main_angle"
    """Column name for storing the main angle of the building."""

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],  # noqa: ARG002
    ) -> GeoDataFrame:
        polygon_buildings_gdf, point_buildings_gdf = (
            data.loc[data.geometry.type == "Polygon"].copy(),
            data.loc[data.geometry.type == "Point"].copy(),
        )
        polygon_buildings_gdf = self._add_attributes_for_area_and_angle(
            polygon_buildings_gdf,
        )
        point_buildings_gdf = self._add_attributes_for_area_and_angle(
            point_buildings_gdf,
        )

        polygon_buildings_gdf = self._filter_buildings_by_area_and_class(
            polygon_buildings_gdf,
        )
        point_buildings_gdf = self._filter_buildings_by_area_and_class(
            point_buildings_gdf,
        )

        polygon_buildings_gdf = self._dissolve_touching_buildings(
            polygon_buildings_gdf,
        )

        always_point_buildings = polygon_buildings_gdf[self.building_class_column].isin(
            self.find_classes_for_point_buildings(),
        )

        # Classify only those buildings that are not flagged as "always_point_buildings"
        classified_gdfs = classify_polygons_by_size_of_minimum_bounding_rectangle(
            polygon_buildings_gdf[~always_point_buildings],
            self.side_threshold,
        )

        # Create centroids for small and large polygons
        classified_gdfs["small_polygons"] = concat(
            [
                classified_gdfs["small_polygons"],
                polygon_buildings_gdf[always_point_buildings],
            ]
        )

        small_building_centroids_gdf = classified_gdfs["small_polygons"].copy()
        small_building_centroids_gdf.geometry = small_building_centroids_gdf.centroid

        large_building_centroids_gdf = classified_gdfs["large_polygons"].copy()
        large_building_centroids_gdf.geometry = large_building_centroids_gdf.centroid

        # Combine original point buildings with centroids of small polygons
        small_buildings_gdf = concat(
            [point_buildings_gdf, small_building_centroids_gdf],
        )

        # Combine centroids of large polygons with all small building points
        all_building_centroids_gdf = GeoDataFrame(
            concat(
                [large_building_centroids_gdf, small_buildings_gdf],
            ),
            crs=point_buildings_gdf.crs,
        )

        small_buildings_gdf = self._generalize_point_buildings(
            small_buildings_gdf,
            all_building_centroids_gdf,
        )

        large_buildings_gdf = self._generalize_polygon_buildings(
            classified_gdfs["large_polygons"],
        )

        result = concat(
            [
                small_buildings_gdf,
                large_buildings_gdf,
            ]
        )

        result.loc[result["old_ids"].isna(), "old_ids"] = None

        return hash_index_from_old_ids(result, "buildings", "old_ids")

    def _generalize_point_buildings(
        self,
        point_buildings_gdf: GeoDataFrame,
        all_buildings_gdf: GeoDataFrame,
    ) -> GeoDataFrame:
        """Generalize points by reducing density based on proximity and priority.

        Args:
        ----
            point_buildings_gdf: A GeoDataFrame containing the centroids of
                small buildings
            all_buildings_gdf: A GeoDataFrame containing the centroids of all buildings

        Returns:
        -------
            A GeoDataFrame containing generalized point buildings.

        """
        # Split buildings into low-priority and others (excluding always-kept)
        low_priority_buildings_gdf = point_buildings_gdf[
            point_buildings_gdf[self.building_class_column].isin(
                self.find_classes_for_low_priority_buildings()
            )
            & ~point_buildings_gdf[self.building_class_column].isin(
                self.find_classes_for_always_kept_buildings(),
            )
        ]

        other_buildings_gdf = point_buildings_gdf[
            ~point_buildings_gdf[self.building_class_column].isin(
                self.find_classes_for_low_priority_buildings()
            )
            & ~point_buildings_gdf[self.building_class_column].isin(
                self.find_classes_for_always_kept_buildings(),
            )
        ]

        # Reduce density separately for each group
        low_priority_buildings_gdf = reduce_nearby_points_by_selecting(
            low_priority_buildings_gdf,
            all_buildings_gdf,
            self.minimum_distance_to_isolated_building,
            self.original_area_column,
        )

        other_buildings_gdf = reduce_nearby_points_by_selecting(
            other_buildings_gdf,
            all_buildings_gdf,
            self.point_size * 1.66,
            self.original_area_column,
        )

        return concat(
            [low_priority_buildings_gdf, other_buildings_gdf],
        )

    def _generalize_polygon_buildings(
        self,
        input_gdf: GeoDataFrame,
    ) -> GeoDataFrame:
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
        if not check_gdf_geometry_type(input_gdf, ["Polygon", "MultiPolygon"]):
            msg = (
                "generalize_polygon_buildings only supports "
                + "Polygon or MultiPolygon geometries."
            )
            raise GeometryTypeError(msg)

        result_gdf = input_gdf.copy()

        # Simplify polygons using CartaGen Ruas simplification
        result_gdf = GeneralizeBuildings._simplify_buildings(
            result_gdf,
            SIMPLIFY_EDGE_THRESHOLD,
        )
        simplified_gdf = result_gdf.copy()

        # Buffer the narrow building parts
        narrow_parts_gdf = extract_narrow_polygon_parts(
            result_gdf,
            self.point_size,
        )

        narrow_parts_gdf.geometry = narrow_parts_gdf.geometry.buffer(
            (BUFFER_SIZE_FOR_NARROW_PARTS), cap_style="flat", join_style="mitre"
        )

        # Dissolve the expanded narrow parts and already large enough parts
        result_gdf = concat([simplified_gdf, narrow_parts_gdf])
        result_gdf["__temp_id"] = result_gdf.index
        result_gdf = dissolve_and_inherit_attributes(
            result_gdf,
            "__temp_id",
        )
        result_gdf = result_gdf.drop("__temp_id", axis=1)

        # Simplify polygons again using CartaGen Ruas simplification
        result_gdf = GeneralizeBuildings._simplify_buildings(
            result_gdf,
            SIMPLIFY_EDGE_THRESHOLD_AFTER_NARROW_PARTS,
        )

        # Subtract buildings from the background
        result_gdf = drop_empty_geometries(result_gdf)
        result_gdf = fix_invalid_geometries(result_gdf)
        simplified_gdf = drop_empty_geometries(simplified_gdf)
        simplified_gdf = fix_invalid_geometries(simplified_gdf)
        bounding_polygon = simplified_gdf.union_all().envelope
        difference_gdf = GeoDataFrame(
            geometry=[bounding_polygon.difference(result_gdf.union_all())],
            crs=input_gdf.crs,
        )

        # Find areas where buildings are less than NARROW_GAPS_THRESHOLD meters apart
        narrow_gaps_gdf = extract_narrow_polygon_parts(
            difference_gdf,
            NARROW_GAPS_THRESHOLD,
        )

        # Remove areas from narrow gaps where actual buildings exist —
        # do not want to widen gaps that already contain building
        narrow_gaps_gdf.geometry = narrow_gaps_gdf.geometry.difference(
            input_gdf.union_all()
        )

        # Expand narrow gaps
        narrow_gaps_gdf = narrow_gaps_gdf.geometry.buffer(
            BUFFER_SIZE_FOR_NARROW_GAPS,
            cap_style="flat",
            join_style="mitre",
        )

        # Subtract expanded gaps from the generalized buildings
        result_gdf.geometry = result_gdf.geometry.difference(
            narrow_gaps_gdf.union_all(),
        )

        # Dissolve buildings with the same building class
        result_gdf = dissolve_and_inherit_attributes(
            result_gdf,
            self.building_class_column,
        )

        # Simplify polygons once again
        result_gdf = GeneralizeBuildings._simplify_buildings(
            result_gdf,
            SIMPLIFY_EDGE_THRESHOLD_AFTER_NARROW_GAPS,
        )

        # Remove holes smaller than the hole_threshold size
        result_gdf = remove_small_holes(
            result_gdf,
            self.hole_threshold,
        )

        # Return buildings larger than a square with side length equal to point_size
        return remove_small_polygons(
            result_gdf,
            self.point_size**2,
        )

    def _dissolve_touching_buildings(
        self,
        input_gdf: GeoDataFrame,
    ) -> GeoDataFrame:
        """Slightly buffer buildings and dissolve adjacent buildings of the same class.

        Returns
        -------
            A GeoDataFrame containing dissolved polygon buildings.

        """
        input_gdf.geometry = input_gdf.buffer(0.1, cap_style="flat", join_style="mitre")
        input_gdf = dissolve_and_inherit_attributes(
            input_gdf,
            self.building_class_column,
        )
        input_gdf.geometry = input_gdf.buffer(
            -0.1,
            cap_style="flat",
            join_style="mitre",
        )

        return input_gdf

    def _filter_buildings_by_area_and_class(
        self,
        input_gdf: GeoDataFrame,
    ) -> GeoDataFrame:
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
                    input_gdf[self.original_area_column]
                    >= self.area_threshold_for_all_buildings
                )
                & ~(
                    (
                        input_gdf[self.building_class_column].isin(
                            self.find_classes_for_low_priority_buildings()
                        )
                    )
                    & (
                        input_gdf[self.original_area_column]
                        < self.area_threshold_for_low_priority_buildings
                    )
                )
            )
            | input_gdf[self.building_class_column].isin(
                self.find_classes_for_always_kept_buildings()
            )
        ]

    def _add_attributes_for_area_and_angle(
        self,
        input_gdf: GeoDataFrame,
    ) -> GeoDataFrame:
        """Ensure every building geometry has attributes for original area and angle.

        Returns
        -------
            A GeoDataFrame with columns for original area and main angle.

        """
        if input_gdf.empty:
            if self.original_area_column not in input_gdf.columns:
                input_gdf[self.original_area_column] = Series(dtype=float)
            if self.main_angle_column not in input_gdf.columns:
                input_gdf[self.main_angle_column] = Series(dtype=float)
            return input_gdf

        geom_types = input_gdf.geometry.geom_type.unique()
        if all(geom_type in {"Polygon", "MultiPolygon"} for geom_type in geom_types):
            if self.original_area_column not in input_gdf.columns:
                input_gdf[self.original_area_column] = input_gdf.geometry.area
            if self.main_angle_column not in input_gdf.columns:
                input_gdf[self.main_angle_column] = input_gdf.geometry.apply(
                    calculate_main_angle
                )

        # If the area and angle attributes of a point building have been lost, they are
        # replaced with 0.0
        elif all(geom_type in {"Point", "MultiPoint"} for geom_type in geom_types):
            if self.original_area_column not in input_gdf.columns:
                input_gdf[self.original_area_column] = 0.0
            if self.main_angle_column not in input_gdf.columns:
                input_gdf[self.main_angle_column] = 0.0
                input_gdf[self.main_angle_column] = Series(dtype=float)

        return input_gdf

    @staticmethod
    def _simplify_buildings(
        input_gdf: GeoDataFrame,
        edge_threshold: float,
    ) -> GeoDataFrame:
        """Simplify geometries using the Cartagen Ruas simplification algorithm.

        Args:
        ----
            input_gdf: A GeoDataFrame containing Polygon or MultiPolygon geometries
            edge_threshold: Minimum edge length to preserve. Edges shorter than this
                  threshold will be simplified.

        Returns:
        -------
            A GeoDataFrame containing simplified polygon geometries.

        Raises:
        ------
            GeometryTypeError: If the input GeoDataFrame contains other than
                  Polygon or MultiPolygon geometries.

        """
        if not check_gdf_geometry_type(input_gdf, ["Polygon", "MultiPolygon"]):
            msg = "Simplify buildings only supports Polygon or MultiPolygon geometries."
            raise GeometryTypeError(msg)

        result_gdf = input_gdf.copy().explode()
        result_gdf.geometry = result_gdf.geometry.apply(
            lambda geom: buildings.simplify_building(geom, edge_threshold),
        )

        return result_gdf

    def find_classes_for_low_priority_buildings(self) -> list[int | str]:
        """Getter function for classes_for_low_priority_buildings.

        Returns:
            The requested attribute, or an empty list if the attribute is None.

        """
        return (
            []
            if self.classes_for_low_priority_buildings is None
            else self.classes_for_low_priority_buildings
        )

    def find_classes_for_point_buildings(self) -> list[int | str]:
        """Getter function for classes_for_point_buildings.

        Returns:
            The requested attribute, or an empty list if the attribute is None.

        """
        return (
            []
            if self.classes_for_point_buildings is None
            else self.classes_for_point_buildings
        )

    def find_classes_for_always_kept_buildings(self) -> list[int | str]:
        """Getter function for classes_for_always_kept_buildings.

        Returns:
            The requested attribute, or an empty list if the attribute is None.

        """
        return (
            []
            if self.classes_for_always_kept_buildings is None
            else self.classes_for_always_kept_buildings
        )
