#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from typing import ClassVar, cast, override

from geopandas import GeoDataFrame
from pandas import concat
from pygeoops import centerline
from shapely.geometry import Polygon

from geogenalg.analyze import get_polygons_for_parallel_lines
from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.attributes import inherit_attributes_from_largest
from geogenalg.continuity import (
    connect_lines_to_polygon_centroids,
    flag_polygon_centerline_connections,
    process_lines_and_reconnect,
)
from geogenalg.core.exceptions import MissingReferenceError
from geogenalg.core.geometry import (
    LineExtendFrom,
    assign_nearest_z,
    extend_line_to_nearest,
)
from geogenalg.identity import hash_index_from_old_ids
from geogenalg.merge import dissolve_and_inherit_attributes
from geogenalg.selection import (
    remove_close_line_segments,
    remove_disconnected_short_lines,
)
from geogenalg.split import explode_and_hash_id


@supports_identity
@dataclass(frozen=True)
class GeneralizePowerLines(BaseAlgorithm):
    """Generalize lines representing power lines.

    The algorithm does the following steps:
    - Merges parallel power lines
    - Ensures network connectivity with power stations
    - Removes short lines
    - Removes lower priority lines which are close to higher priority lines
    - Simplifies lines
    """

    distance_threshold_for_parallel_lines: float = 50.0
    """Minimum distance for lines to be considered to be parallel."""
    classes_for_merge_parallel_lines: list[int | str] = field(default_factory=list)
    """Attribute values to distinguish which types of parallel lines will be merged."""
    classes_for_higher_priority_lines: list[int | str] = field(default_factory=list)
    """Attribute values to distinguish which types of lines will be kept if
    lower priority lines run in parallel."""
    class_column: str = "function"
    """Column in which attribute values to distinguish power line types reside."""
    length_threshold: float = 100.0
    """Minimum length of a power line."""
    simplification_tolerance: float = 0.0
    """Tolerance used for geometry simplification."""
    reference_key_fences: str = "fences"
    """Reference key for generalized fence dataset."""
    reference_key_substations: str = "substations"
    """Reference key for substation dataset."""

    valid_input_geometry_types: ClassVar = {"LineString"}
    valid_reference_geometry_types: ClassVar = {"LineString", "Polygon"}

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        if self.reference_key_fences not in reference_data:
            raise MissingReferenceError

        if self.reference_key_substations not in reference_data:
            raise MissingReferenceError

        fences = reference_data[self.reference_key_fences]
        substations = reference_data[self.reference_key_substations]

        # Get substations which no longer have a surrounding fence
        substations = substations.loc[
            substations.geometry.buffer(5).disjoint(fences.union_all())
        ]

        gdf = cast("GeoDataFrame", data.copy())

        if self.classes_for_merge_parallel_lines:
            merged = self._merge_parallel_lines(
                gdf.loc[
                    gdf[self.class_column].isin(self.classes_for_merge_parallel_lines)
                ],
                fences,
                substations,
            )
            gdf = concat(
                [
                    gdf.loc[
                        ~gdf[self.class_column].isin(
                            self.classes_for_merge_parallel_lines
                        )
                    ],
                    merged,
                ],
            )

        if self.classes_for_higher_priority_lines:
            higher_priority_lines = gdf.loc[
                gdf[self.class_column].isin(self.classes_for_higher_priority_lines)
            ].copy()
            lower_priority_lines = gdf.loc[
                ~gdf[self.class_column].isin(self.classes_for_higher_priority_lines)
            ].copy()

            def _process(lower_gdf: GeoDataFrame) -> GeoDataFrame:
                return explode_and_hash_id(
                    remove_close_line_segments(
                        lower_gdf,
                        higher_priority_lines,
                        self.distance_threshold_for_parallel_lines,
                    ),
                    "powerlines",
                )

            lower_priority_lines = process_lines_and_reconnect(
                lower_priority_lines,
                _process,
                higher_priority_lines,
            )

            gdf = concat([higher_priority_lines, lower_priority_lines])

        if self.simplification_tolerance > 0:
            gdf.geometry = gdf.geometry.simplify(self.simplification_tolerance)

        gdf = remove_disconnected_short_lines(gdf, self.length_threshold)
        gdf = connect_lines_to_polygon_centroids(gdf, substations)

        return assign_nearest_z(data, gdf)

    def _merge_parallel_lines(
        self,
        data: GeoDataFrame,
        fences: GeoDataFrame,
        substations: GeoDataFrame,
    ) -> GeoDataFrame:
        gdf = data.copy()

        parallels = get_polygons_for_parallel_lines(
            gdf.loc[gdf[self.class_column].isin(self.classes_for_merge_parallel_lines)],
            self.distance_threshold_for_parallel_lines,
            allowed_direction_difference=10,
            polygonize_function="convex",
        )

        if parallels.empty:
            return gdf

        # Remove holes
        parallels.geometry = parallels.geometry.apply(
            lambda geom: Polygon(geom.exterior)
        )

        parallels = parallels.set_index(parallels.index.astype("string"))
        parallels = inherit_attributes_from_largest(
            gdf.loc[gdf[self.class_column].isin(self.classes_for_merge_parallel_lines)],
            parallels,
            old_ids_column="old_ids",
            measure_by="length",
        )
        parallels = dissolve_and_inherit_attributes(parallels, old_ids_column="old_ids")
        parallels = hash_index_from_old_ids(parallels, "powerlines", "old_ids")

        parallels["temp_polygon"] = parallels.geometry
        parallels.geometry = parallels.geometry.segmentize(5)
        parallels.geometry = parallels.geometry.apply(
            lambda geom: centerline(
                geom,
                simplifytolerance=0.5,
                min_branch_length=self.length_threshold / 2,
            ),
        )

        parallels = parallels.explode()

        fences_union = fences.union_all()

        start_connected_column = "__start_connected"
        end_connected_column = "__end_connected"
        parallels = flag_polygon_centerline_connections(
            parallels,
            fences,
            "temp_polygon",
        )
        parallels.geometry = parallels[
            [start_connected_column, end_connected_column, parallels.geometry.name]
        ].apply(
            lambda columns: extend_line_to_nearest(
                columns[parallels.geometry.name],
                fences_union,
                LineExtendFrom.from_bools(
                    extend_start=columns[start_connected_column],
                    extend_end=columns[end_connected_column],
                ),
            ),
            axis=1,
        )

        subs_union = substations.union_all()
        parallels = flag_polygon_centerline_connections(
            parallels,
            substations,
            "temp_polygon",
        )
        parallels.geometry = parallels[
            [start_connected_column, end_connected_column, parallels.geometry.name]
        ].apply(
            lambda columns: extend_line_to_nearest(
                columns[parallels.geometry.name],
                subs_union,
                LineExtendFrom.from_bools(
                    extend_start=columns[start_connected_column],
                    extend_end=columns[end_connected_column],
                ),
                self.distance_threshold_for_parallel_lines,
            ),
            axis=1,
        )

        def _process(gdf: GeoDataFrame) -> GeoDataFrame:
            # Remove lines from areas where parallel lines have been merged
            # together. Not all (smaller) parallel polygons form a centerline.
            # Therefore when removing parallel lines from areas, remove only
            # where centerline was formed.
            parallel_polygons_union = parallels["temp_polygon"].union_all()
            parallel_overlay = parallels.loc[
                parallels.geometry.intersects(parallel_polygons_union)
            ]
            parallel_overlay_union = (
                parallel_overlay["temp_polygon"].buffer(1).union_all()
            )
            gdf.geometry = gdf.geometry.difference(parallel_overlay_union)
            gdf = gdf.loc[~gdf.geometry.is_empty]
            return explode_and_hash_id(gdf, "powerlines")

        gdf = process_lines_and_reconnect(
            gdf,
            _process,
            reconnect_to=parallels,
        )

        parallels = parallels.drop("temp_polygon", axis=1)

        combined = concat([gdf, parallels])
        combined = combined.drop([start_connected_column, end_connected_column], axis=1)

        return explode_and_hash_id(combined, "powerlines")
