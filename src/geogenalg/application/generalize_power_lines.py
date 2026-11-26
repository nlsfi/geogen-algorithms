#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import ClassVar, override

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pydantic import Field
from shapely import (
    buffer,
    difference,
    get_parts,
    intersection,
    length,
    simplify,
)
from shapely.geometry import LineString, MultiLineString, Polygon
from shapelysmooth import taubin_smooth

from geogenalg.analyze import polygonize_parallel_lines
from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.attributes import inherit_attributes_from_largest
from geogenalg.continuity import (
    add_contiguous_lines_information,
    connect_lines_to_polygon_centroids,
    flag_polygon_centerline_connections,
    get_lines_along_reference_lines,
    process_lines_and_reconnect,
)
from geogenalg.core.geometry import (
    assign_nearest_z,
    extend_lines_to_nearest,
    union_by_predicate,
)
from geogenalg.core.parallel import in_parallel
from geogenalg.identity import hash_index_from_old_ids
from geogenalg.merge import dissolve_and_inherit_attributes
from geogenalg.split import explode_and_hash_id
from geogenalg.utility.dataframe_processing import combine_gdfs

_SUBSTATION_FENCE_LENGTH_RATIO_THRESHOLD = 0.75
_FENCE_CONNECTION_TOLERANCE = 5


def _smooth_keeping_ends(
    lines: LineString | MultiLineString,
    steps: int,
) -> LineString | MultiLineString:
    smoothed = []

    for part in get_parts(lines):
        coords = list(part.coords)

        if len(coords) < 3:  # noqa: PLR2004
            smoothed.append(part)
        elif part.is_closed:
            middle = len(coords) // 2

            first = taubin_smooth(LineString(coords[: middle + 1]), steps=steps)
            second = taubin_smooth(LineString(coords[middle:]), steps=steps)
            smoothed.append(LineString([*first.coords, *second.coords[1:]]))
        else:
            smoothed.append(taubin_smooth(part, steps=steps))

    if len(smoothed) == 1:
        return smoothed[0]

    return MultiLineString(smoothed)


@supports_identity
class GeneralizePowerLines(BaseAlgorithm):
    """Generalize lines representing power lines.

    The algorithm does the following steps:
    - Merges parallel power lines
    - Ensures network connectivity with power stations
    - Removes short lines
    - Removes lower priority lines which are close to higher priority lines
    - Simplifies lines
    """

    distance_threshold_for_parallel_lines: float = Field(50.0, gt=0)
    """Minimum distance for lines to be considered to be parallel."""
    parallel_line_allowed_direction_difference: float = Field(10.0, ge=0, le=90)
    """Maximum difference in direction, for lines to be considered parallel (in
    degrees)."""
    classes_for_merge_parallel_lines: frozenset[int | str] = frozenset()
    """Attribute values to distinguish which types of parallel lines will be merged."""
    classes_for_higher_priority_lines: frozenset[int | str] = frozenset()
    """Attribute values to distinguish which types of lines will be kept if
    lower priority lines run in parallel."""
    lower_priority_minimum_percentage: float = Field(90.0, ge=0.0, le=100.0)
    """A lower priority line is removed if at least this percentage of its
    length is within distance_threshold_for_parallel_lines of higher priority
    lines."""
    class_column: str = "function"
    """Column in which attribute values to distinguish power line types reside."""
    length_threshold: float = Field(100.0, gt=0)
    """Minimum length of a power line."""
    simplification_tolerance: float = Field(0.0, ge=0.0)
    """Tolerance used for geometry simplification."""
    centerline_segmentize_distance: float = Field(10.0, gt=0)
    """Segmentization distance used when building centerline for merged parallel lines.
    Lower value gives better result, but is slower."""
    centerline_smoothing_steps: int = Field(5, ge=0)
    """Number of smoothing passes applied to the centerline of merged parallel
    lines."""
    centerline_merge_vertices_distance: float = Field(0.1, ge=0)
    """Vertexes closer than this value of polygonized parallel lines will be
    removed. This can prevent small disconnected lines from forming in the
    centerline."""
    substation_connection_tolerance: float = Field(0.1, ge=0)
    """Distance tolerance for a line to still be considered touching a substation."""
    maximum_extension_length: float = Field(25.0, gt=0)
    """Maximum length by which a merged parallel line is extended to connect
    it to a fence or substation."""
    maximum_extension_angle: float = Field(90.0, gt=0, le=180)
    """Maximum angle in degrees between the direction of a merged parallel
    line at its end and the extension connecting it to a fence or substation."""
    reference_key_fences: str = "fences"
    """Reference key for fence dataset. This mandatory linestring reference
    data is intended to contain fences which wrap around the substations
    reference data. The data is intended to already be generalized to the
    target scale."""
    reference_key_substations: str = "substations"
    """Reference key for substation dataset. This mandatory reference data is
    intended to contain polygons around which reference fence features may wrap
    around. The data has to be of the previous scale. This is because the data
    is used in conjuction with the fence data to identify substations which no
    longer have a fence feature around them. In this case a power line
    connecting to a substation will be extended to the substation's
    centroid."""

    valid_input_geometry_types: ClassVar = {"LineString"}
    reference_data_schema: ClassVar = {
        "reference_key_fences": ReferenceDataInformation(
            required=True,
            valid_geometry_types={
                "LineString",
            },
        ),
        "reference_key_substations": ReferenceDataInformation(
            required=True,
            valid_geometry_types={
                "Polygon",
            },
        ),
    }

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        fences = reference_data[self.reference_key_fences]
        substations = reference_data[self.reference_key_substations]

        # Get substations which no longer have a surrounding fence.

        # Since the fence geometry may have changed (been simplified), we do
        # this by buffering the substation exterior and calculating the
        # fraction of the length found in all fences compared to the exterior
        # length. This way if other fences happen to be close to the substation
        # they are unlikely going to affect the result in an unwanted way.
        exterior_areas = substations.geometry.exterior.buffer(
            _FENCE_CONNECTION_TOLERANCE
        ).to_numpy()
        nearby_fences = union_by_predicate(
            exterior_areas,
            fences.geometry.to_numpy(),
            "intersects",
        )
        fence_lengths = length(intersection(exterior_areas, nearby_fences))
        substations = substations.loc[
            fence_lengths / substations.geometry.exterior.length.to_numpy()
            <= _SUBSTATION_FENCE_LENGTH_RATIO_THRESHOLD
        ]

        gdf = data.copy()

        if self.classes_for_merge_parallel_lines:
            merged = self._merge_parallel_lines(
                gdf.loc[
                    gdf[self.class_column].isin(self.classes_for_merge_parallel_lines)
                ],
                fences,
                substations,
            )
            gdf = combine_gdfs(
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
                return get_lines_along_reference_lines(
                    lower_gdf,
                    higher_priority_lines,
                    self.distance_threshold_for_parallel_lines,
                    length_percentage=self.lower_priority_minimum_percentage,
                )[1]

            lower_priority_lines = process_lines_and_reconnect(
                lower_priority_lines,
                _process,
                higher_priority_lines,
                length_tolerance=self.distance_threshold_for_parallel_lines * 1.25,
                cut_tolerance=0.0,
            )

            gdf = combine_gdfs([higher_priority_lines, lower_priority_lines])

        if self.simplification_tolerance > 0:
            gdf.geometry = gdf.geometry.simplify(self.simplification_tolerance)

        gdf = add_contiguous_lines_information(
            gdf,
            GeoDataFrame(
                geometry=fences.geometry.buffer(_FENCE_CONNECTION_TOLERANCE),
                crs=fences.crs,
            ),
        )
        gdf = GeoDataFrame(
            gdf.loc[
                ~(
                    (gdf["contiguous_length"] <= self.length_threshold)
                    & (gdf["contiguous_dead_end"] | gdf["contiguous_disconnected"])
                )
            ].drop(
                columns=[
                    "contiguous_length",
                    "contiguous_dead_end",
                    "contiguous_disconnected",
                ]
            ),
            geometry=gdf.geometry.name,
            crs=gdf.crs,
        )
        gdf = connect_lines_to_polygon_centroids(
            gdf,
            substations,
            connection_tolerance=self.substation_connection_tolerance,
        )

        return assign_nearest_z(data, gdf, overwrite_z=True)

    def _merge_parallel_lines(
        self,
        data: GeoDataFrame,
        fences: GeoDataFrame,
        substations: GeoDataFrame,
    ) -> GeoDataFrame:
        gdf = data.copy()

        parallels = polygonize_parallel_lines(
            gdf.loc[gdf[self.class_column].isin(self.classes_for_merge_parallel_lines)],
            self.distance_threshold_for_parallel_lines,
            maximum_angle_difference=self.parallel_line_allowed_direction_difference,
        )

        if parallels.empty:
            return gdf

        # Remove holes
        parallels.geometry = parallels.geometry.apply(
            lambda geom: Polygon(geom.exterior)
        )  # ty: ignore[no-matching-overload]

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

        centerlines = (
            in_parallel(parallels.geometry)
            .centerline(
                simplify_tolerance=0,
                prune_branches_shorter_than=self.length_threshold / 2,
                segmentize_by=self.centerline_segmentize_distance,
                merge_vertices_closer_than=self.centerline_merge_vertices_distance,
            )
            .resolve_numpy()
        )

        # It can be faster to form the centerline with a higher densification distance.
        # However, this can introduce jagged lines. Smoothing helps with those.
        smoothed = np.empty(len(centerlines), dtype=object)
        smoothed[:] = [
            _smooth_keeping_ends(line, steps=self.centerline_smoothing_steps)
            if self.centerline_smoothing_steps > 0
            else line
            for line in centerlines
        ]
        parallels.geometry = GeoSeries(
            simplify(smoothed, 0.5),
            index=parallels.index,
            crs=parallels.crs,
        )

        parallels = parallels.explode(column=parallels.geometry.name)

        start_connected_column = "_start_connected"
        end_connected_column = "_end_connected"

        def _extend_connected_ends(
            lines: GeoDataFrame,
            extend_to: np.ndarray,
        ) -> GeoSeries:
            return GeoSeries(
                extend_lines_to_nearest(
                    lines.geometry.to_numpy(),
                    extend_to,
                    extend_start=lines[start_connected_column].to_numpy(),
                    extend_end=lines[end_connected_column].to_numpy(),
                    length_tolerance=self.maximum_extension_length,
                    maximum_turn_angle=self.maximum_extension_angle,
                ),
                index=lines.index,
                crs=lines.crs,
            )

        parallels = flag_polygon_centerline_connections(
            parallels,
            fences,
            "temp_polygon",
        )
        parallels.geometry = _extend_connected_ends(
            parallels,
            fences.geometry.to_numpy(),
        )

        parallels = flag_polygon_centerline_connections(
            parallels,
            substations,
            "temp_polygon",
            connection_tolerance=self.substation_connection_tolerance,
        )
        parallels.geometry = _extend_connected_ends(
            parallels,
            substations.geometry.to_numpy(),
        )

        def _process(gdf: GeoDataFrame) -> GeoDataFrame:
            # Remove lines from areas where parallel lines have been merged
            # together. Not all (smaller) parallel polygons form a centerline.
            # Therefore when removing parallel lines from areas, remove only
            # where centerline was formed.
            polygons = parallels.loc[~parallels.geometry.is_empty, "temp_polygon"]
            removal_areas = buffer(polygons[~polygons.duplicated()].to_numpy(), 1)
            lines = gdf.geometry.to_numpy()
            nearby_areas = union_by_predicate(lines, removal_areas, "intersects")
            gdf.geometry = GeoSeries(
                difference(lines, nearby_areas),
                index=gdf.index,
                crs=gdf.crs,
            )
            gdf = gdf.loc[~gdf.geometry.is_empty]
            return explode_and_hash_id(gdf, "powerlines")

        gdf = process_lines_and_reconnect(
            gdf,
            _process,
            reconnect_to=parallels,
        )

        parallels = parallels.drop("temp_polygon", axis=1)

        combined = combine_gdfs([gdf, parallels])
        combined = combined.drop([start_connected_column, end_connected_column], axis=1)

        return explode_and_hash_id(combined, "powerlines")
