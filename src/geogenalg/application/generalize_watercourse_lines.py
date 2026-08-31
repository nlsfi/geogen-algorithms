#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from cartagen.enrichment.network.strokes import strokes_rivers, strokes_roads
from geogenalg.continuity import add_contiguous_lines_information, count_connections
from functools import partial
from geogenalg.attributes import inherit_attributes_for_lines_by_buffer
from geogenalg.utility.dataframe_processing import copy_gdf_as_empty, combine_gdfs
from typing import ClassVar, cast

from geopandas.geodataframe import GeoDataFrame
from pydantic import Field

from geogenalg.analyze import  flag_parallel_lines, flag_parallel_lines2
from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.geometry import sinuosity, mean_segment_length, line_segment_direction_std_dev, split_at_direction_changes
from geogenalg.selection import rank_parallel_lines, remove_nth_ranks


@supports_identity
class GeneralizeWatercourseLines(BaseAlgorithm):
    """..."""

    parallel_line_distance: float = Field(45.0, ge=0.0)
    """..."""
    parallel_line_keep_frequency: int = Field(3, gt=0)
    """..."""
    parallel_line_keep_edges_always: bool = True
    """..."""
    natural_max_mean_segment_length: float = Field(5.0, gt=0.0)
    """..."""
    natural_min_vertices: int = Field(10, gt=0)
    """..."""
    threshold_length: float = Field(75.0, ge=0)
    """Unconnected/dead-end linestring shorter than this will be removed."""

    valid_input_geometry_types: ClassVar = {"LineString"}
    requires_projected_crs: ClassVar = True

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],  # noqa: ARG002
    ) -> GeoDataFrame:
        gdf = data.copy()

        gdf["sinuosity"] = gdf.geometry.apply(sinuosity)
        gdf["mean_segment_length"] = gdf.geometry.apply(mean_segment_length)
        gdf["vertices"] = gdf.geometry.apply(lambda geom: len(geom.coords))
        gdf["likely_artificial"] = (gdf["mean_segment_length"] >= self.natural_max_mean_segment_length) | (gdf["vertices"] < self.natural_min_vertices)
        gdf.to_file("all_with_attributes.gpkg")
        gdf = gdf.loc[gdf["likely_artificial"]]

        gdf.geometry = gdf.geometry.normalize().apply(split_at_direction_changes)

        return gdf

        to_rank = copy_gdf_as_empty(gdf)
        remainder = gdf

        minimum_lines_in_a_group = 4

        # FIXME: ensure this has guaranteed end condition
        i = 0
        while not remainder.empty:
            flagged = flag_parallel_lines2(
                remainder,
                self.parallel_line_distance,
                allowed_direction_difference=10.0,
            )

            flagged = flagged.dissolve(by="parallel_group", as_index=False)
            flagged.geometry = flagged.geometry.line_merge()
            flagged = flagged.explode(reset_index=False).reset_index(drop=True)
            flagged = flagged.loc[flagged.geometry.length > 80]

            for parallel_group in flagged["parallel_group"].unique():
                lines = flagged.loc[flagged["parallel_group"] == parallel_group]
                ranked, non_ranked = rank_parallel_lines(lines)

                flagged.loc[ranked.index, "rank"] = ranked
                flagged.loc[non_ranked, "rank"] = -1

            flagged = flagged.loc[flagged.groupby("parallel_group")["parallel_group"].transform("size") >= minimum_lines_in_a_group]

            remainder = flagged.loc[flagged["rank"] == -1]

            remainder.to_file(f"remainder_{i}.gpkg")

            to_rank = combine_gdfs(
                [
                    to_rank,
                    flagged.loc[(flagged["rank"] != -1)],
                ],
            )

            i += 1
            print(f"ROUND {i}")

        to_rank.to_file("to_rank_before.gpkg")

        to_rank = (
            to_rank.groupby("parallel_group", group_keys=False)[to_rank.columns]
               .apply(
                remove_nth_ranks,
                rank_column="rank",
                n=self.parallel_line_keep_frequency,
                remove_extremes=self.parallel_line_keep_edges_always,
            )
        )

        to_rank.to_file("to_rank_after.gpkg")

        gdf = data.copy()
        gdf = gdf.overlay(
            to_rank,
            how="difference",
        )

        gdf = add_contiguous_lines_information(gdf)

        gdf = gdf.loc[
            ~(
                (gdf["contiguous_length"] <= self.threshold_length)
                & (gdf["contiguous_dead_end"] | gdf["contiguous_disconnected"])
            )
        ]
        return gdf

        return gdf

        gdf.to_file("artificial.gpkg")

        to_rank = copy_gdf_as_empty(gdf)
        remainder = gdf

        minimum_lines_in_a_group = 4

        # FIXME: ensure this has guaranteed end condition
        i = 0
        while not remainder.empty:
            flagged = flag_parallel_lines(
                remainder,
                self.parallel_line_distance,
                allowed_direction_difference=5.0,
            )

            flagged = flagged.dissolve(by="parallel_group", as_index=False)
            flagged.geometry = flagged.geometry.line_merge()
            flagged = flagged.explode(reset_index=False).reset_index(drop=True)
            flagged = flagged.loc[flagged.geometry.length > 80]

            for parallel_group in flagged["parallel_group"].unique():
                lines = flagged.loc[flagged["parallel_group"] == parallel_group]
                ranked, non_ranked = rank_parallel_lines(lines)

                flagged.loc[ranked.index, "rank"] = ranked
                flagged.loc[non_ranked, "rank"] = -1

            flagged = flagged.loc[flagged.groupby("parallel_group")["parallel_group"].transform("size") >= minimum_lines_in_a_group]

            remainder = flagged.loc[flagged["rank"] == -1]

            remainder.to_file(f"remainder_{i}.gpkg")

            to_rank = combine_gdfs(
                [
                    to_rank,
                    flagged.loc[(flagged["rank"] != -1)],
                ],
            )

            i += 1
            print(f"ROUND {i}")

        to_rank.to_file("to_rank_before.gpkg")

        to_rank = (
            to_rank.groupby("parallel_group", group_keys=False)[to_rank.columns]
               .apply(
                remove_nth_ranks,
                rank_column="rank",
                n=self.parallel_line_keep_frequency,
                remove_extremes=self.parallel_line_keep_edges_always,
            )
        )

        to_rank.to_file("to_rank_after.gpkg")

        gdf = data.copy()
        gdf = gdf.overlay(
            to_rank,
            how="difference",
        )

        gdf = add_contiguous_lines_information(gdf)

        gdf = gdf.loc[
            ~(
                (gdf["contiguous_length"] <= self.threshold_length)
                & (gdf["contiguous_dead_end"] | gdf["contiguous_disconnected"])
            )
        ]
        return gdf
