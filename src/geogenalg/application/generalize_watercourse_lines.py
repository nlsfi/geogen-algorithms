#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import ClassVar, override

import numpy as np
from geopandas.geodataframe import GeoDataFrame
from pydantic import Field
from shapely import (
    centroid,
    get_coordinates,
    get_num_coordinates,
    intersects,
    union_all,
)
from shapely.geometry import LineString

from geogenalg.analyze import add_parallel_line_information
from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.continuity import (
    filter_short_contiguous_lines,
)
from geogenalg.core.geometry import (
    build_collinear_chains,
    line_length_weighted_directions,
    mean_segment_lengths,
)
from geogenalg.selection import prune_parallel_groups
from geogenalg.utility.dataframe_processing import combine_gdfs, copy_gdf_as_empty


@supports_identity
class GeneralizeWaterCourseLines(BaseAlgorithm):
    """Generalize linear watercourse lines.

    Input should contain LineString geometries.

    Reference data may contain polygonal water area geometries, to which the input
    lines may be connected to.

    The algorithm does the following steps:
        - Identifies areas where there are parallel watercourse lines
        - Groups parallel watercourse lines
        - Prunes them, increasing the average distance between the parallel lines
        - Removes disconnected and dead end lines which are too short
    """

    parallel_line_distance: float = Field(45.0, ge=0.0)
    """Minimum distance for lines to be considered parallel."""
    parallel_line_allowed_direction_difference: float = Field(10.0, ge=0.0)
    """Maximum angular difference for lines to be considered parallel or collinear."""
    parallel_line_keep_edges_always: bool = True
    """Controls whether the edges of parallel line groups should always be kept."""
    parallel_line_min_overlap_ratio: float = Field(0.75, ge=0.0, le=1.0)
    """Lines have to parallerly overlap this fraction of their length for them
    to be considered parallel."""
    natural_max_mean_segment_length: float = Field(10.0, gt=0.0)
    """Value used to identify natural features."""
    natural_min_vertices: int = Field(10, gt=0)
    """Value used to identify natural features."""
    disconnected_lines_length_threshold: float = Field(75.0, ge=0)
    """Unconnected/dead-end linestring shorter than this will be removed."""
    prune_distance_multiplier: float = Field(1.25, gt=1.0)
    """Multiplier to establish new approximate distance between parallel lines.
    Higher values result in more pruning."""
    always_keep_column: str | None = None
    """Name of column with values describing features which will never be removed."""
    always_keep_values: frozenset[int | str] = frozenset()
    """Values describing features which will never be removed."""
    reference_key: str = "water_areas"
    """Reference data key for water area data. This optional reference data is
    used to identify whether a watercourse line touches a water area, in which case
    it will not be filtered based on its deadend length. The data is should be in the
    same scale as the input."""

    valid_input_geometry_types: ClassVar = {"LineString"}

    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=False,
            valid_geometry_types={"Polygon"},
        ),
    }

    requires_projected_crs: ClassVar = True

    @staticmethod
    def _refine_parallel_groups(
        gdf: GeoDataFrame,
        parallel_distance: float,
        allowed_direction_difference: float,
        max_iterations: int = 10,
    ) -> GeoDataFrame:
        result = gdf.copy().reset_index(drop=True)
        groups = np.full(len(result), -1, dtype=int)
        next_group = 0

        for _ in range(max_iterations):
            unassigned = np.flatnonzero(groups == -1)
            if len(unassigned) < 2:  # noqa: PLR2004
                break

            flagged = add_parallel_line_information(
                result.iloc[unassigned],
                parallel_distance,
                allowed_direction_difference,
            )
            flagged = flagged[flagged["parallel_group"] != -1]
            if flagged.empty:
                break

            evaluated = (
                GeneralizeWaterCourseLines._evaluate_perpendicular_intersections(
                    flagged,
                )
            )

            all_passed = True
            for _, group in evaluated.groupby("parallel_group"):
                passing = group.index[group["intersects_bisector"]]
                all_passed = all_passed and len(passing) == len(group)

                if len(passing) >= 2:  # noqa: PLR2004
                    groups[passing] = next_group
                    next_group += 1

            if all_passed:
                break

        result["parallel_group"] = groups
        return result

    @staticmethod
    def _evaluate_perpendicular_intersections(gdf: GeoDataFrame) -> GeoDataFrame:  # noqa: PLR0914
        result = gdf.copy()
        flags = np.zeros(len(result), dtype=bool)
        geoms = result.geometry.to_numpy()

        for group_id, positions in result.groupby("parallel_group").indices.items():
            if group_id == -1:
                continue

            group_geoms = geoms[positions]
            directions = line_length_weighted_directions(group_geoms)

            chain_ids, chain_geoms = build_collinear_chains(
                group_geoms, directions, 90.0
            )

            doubled = np.radians(2 * directions)
            perpendicular = (
                0.5 * np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())
                + np.pi / 2
            )

            group_union = union_all(group_geoms)
            center_x, center_y = get_coordinates(centroid(group_union))[0]
            min_x, min_y, max_x, max_y = group_union.bounds
            half_length = 2 * np.hypot(max_x - min_x, max_y - min_y)
            dx = half_length * np.cos(perpendicular)
            dy = half_length * np.sin(perpendicular)

            bisector = LineString(
                [(center_x - dx, center_y - dy), (center_x + dx, center_y + dy)]
            )

            flags[positions] = intersects(chain_geoms, bisector)[chain_ids]

        result["intersects_bisector"] = flags
        return result

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        gdf = data.copy()
        gdf["_index"] = data.index

        gdf["touches_water_area"] = False
        if self.reference_key in reference_data:
            intersecting = gdf.sjoin(
                reference_data[self.reference_key],
            ).index.unique()
            gdf.loc[intersecting, "touches_water_area"] = True

        if self.always_keep_column is not None and self.always_keep_values:
            keep_mask = gdf[self.always_keep_column].isin(self.always_keep_values)
            always_kept = gdf.loc[keep_mask].copy()
            gdf = gdf.loc[~keep_mask].copy()
        else:
            always_kept = copy_gdf_as_empty(gdf)

        geoms = gdf.geometry.to_numpy()
        artificial_mask = (
            mean_segment_lengths(geoms) >= self.natural_max_mean_segment_length
        ) | (get_num_coordinates(geoms) < self.natural_min_vertices)

        gdf["parallels_left"] = 0
        gdf["parallels_right"] = 0

        natural = gdf.loc[~artificial_mask].copy()
        artificial = gdf.loc[artificial_mask].copy()

        if not artificial.empty:
            artificial = add_parallel_line_information(
                artificial,
                self.parallel_line_distance,
                self.parallel_line_allowed_direction_difference,
                min_overlap_ratio=self.parallel_line_min_overlap_ratio,
            )
            artificial = self._refine_parallel_groups(
                artificial,
                self.parallel_line_distance,
                self.parallel_line_allowed_direction_difference,
            )
            artificial, _ = prune_parallel_groups(
                artificial,
                distance_multiplier=self.prune_distance_multiplier,
                keep_furthest=self.parallel_line_keep_edges_always,
            )

        combined = combine_gdfs([artificial, natural], ignore_index=True)

        middle = (
            (combined["parallels_left"] > 0) & (combined["parallels_right"] > 0)
        ).to_numpy()

        keep = filter_short_contiguous_lines(
            combined.geometry.to_numpy(),
            self.disconnected_lines_length_threshold,
            dead_end_exempt=middle | combined["touches_water_area"].to_numpy(),
            disconnected_exempt=middle,
        )
        combined = combined.loc[keep]

        combined = combine_gdfs([combined, always_kept], ignore_index=True)

        result = combined[data.columns].copy()
        result.index = combined["_index"]
        result.index.name = data.index.name

        return result
