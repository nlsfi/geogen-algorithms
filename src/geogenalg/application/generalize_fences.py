#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import override

from cartagen.utils.partitioning.network import network_faces
from geopandas import GeoDataFrame
from pandas import concat

from geogenalg import continuity, merge, selection
from geogenalg.application import BaseAlgorithm
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.validation import check_gdf_geometry_type


@dataclass(frozen=True)
class GeneralizeFences(BaseAlgorithm):
    """Generalize lines representing fences.

    Reference data should contain a Point GeoDataFrame with the key
    "masts".

    Output contains the generalized line fences.

    The algorithm does the following steps:
    - Merges line segments
    - Adds helper lines to close small gaps between lines
    - Removes short lines within large enough enclosed areas
    - Removes surrounding lines of small enough enclosed areas
    - Removes close enough lines surrounding a mast
    - Removes all short lines
    - Simplifies all lines
    """

    closing_fence_area_threshold: float = 2000.0
    """Minimum area for a fence-enclosed region."""
    closing_fence_area_with_mast_threshold: float = 8000.0
    """Minimum area for a fence-enclosed region containing a mast."""
    fence_length_threshold: float = 80.0
    """Minimum length for a fence line."""
    fence_length_threshold_in_closed_area: float = 300.0
    """Minimum length for a fence line within a closed area."""
    simplification_tolerance: float = 4.0
    """Tolerance used for geometry simplification."""
    gap_threshold: float = 25.0
    """Maximum gap between two fence lines to be connected with a helper line."""
    attribute_for_line_merge: str = "kohdeluokka"
    """Name of the attribute to determine which line features can be merged."""

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: A GeoDataFrame containing the fence lines to be generalized.
            reference_data: Should contain a Point GeoDataFrame with the key
                "masts".

        Returns:
        -------
            GeoDataFrame containing the generalized fence lines.

        Raises:
        ------
            GeometryTypeError: If `data` contains non-line geometries or the
                GeoDataFrame with key "masts" in `reference_data` contains
                non-point geometries.
            KeyError: If `reference_data` does not contain data with key
                "masts" or input data does not have specified
                `attribute_for_line_merge`.

        """
        if not check_gdf_geometry_type(data, ["LineString"]):
            msg = "GeneralizeFences works only with LineString geometries."
            raise GeometryTypeError(msg)
        if "masts" not in reference_data:
            msg = (
                "GeneralizeFences requires mast Point GeoDataFrame"
                + " in reference_data with key 'masts'."
            )
            raise KeyError(msg)
        if not check_gdf_geometry_type(reference_data["masts"], ["Point"]):
            msg = "Masts data should be a Point GeoDataFrame."
            raise GeometryTypeError(msg)
        if self.attribute_for_line_merge not in data.columns:
            msg = (
                "Specified `attribute_for_line_merge` "
                + f"({self.attribute_for_line_merge}) not found in input GeoDataFrame."
            )
            raise KeyError(msg)

        result_gdf = data.copy()

        # Merge connecting lines with the same attribute value
        result_gdf = merge.merge_connecting_lines_by_attribute(
            result_gdf, self.attribute_for_line_merge
        )

        # Generate helper lines to close small gaps
        helper_lines_gdf = continuity.connect_nearby_endpoints(
            result_gdf, self.gap_threshold
        )

        # Combine original fence lines with helper lines
        combined_gdf: GeoDataFrame = concat(
            [result_gdf, helper_lines_gdf], ignore_index=True
        )

        # Calculate the CartaGen network faces to fill closing line geometries
        faces = network_faces(list(combined_gdf.geometry), convex_hull=False)
        faces_gdf = GeoDataFrame(geometry=list(faces))

        # Dissolve adjacent polygons into larger contiguous areas
        faces_gdf = faces_gdf.dissolve(as_index=False)
        faces_gdf = faces_gdf.union_all()

        # Select fence lines shorter than the threshold for enclosed areas
        short_lines = result_gdf[
            result_gdf.geometry.length < self.fence_length_threshold_in_closed_area
        ]

        # Remove short fence lines that are within closed polygonal areas
        to_remove_idx = short_lines[
            short_lines.geometry.apply(faces_gdf.contains)
        ].index
        result_gdf = result_gdf.drop(index=to_remove_idx)

        # Convert MultiPolygon result back into a proper GeoDataFrame
        polygons = list(faces_gdf.geoms)
        faces_gdf = GeoDataFrame(geometry=polygons, crs=data.crs)

        # Remove polygons whose area exceeds the closing_fence_area_with_mast_threshold
        # and the closing_fence_area_threshold
        polygon_gdf_with_point, polygon_gdf_without_point = (
            selection.split_polygons_by_point_intersection(
                faces_gdf, reference_data["masts"]
            )
        )
        polygon_gdf_with_point = selection.remove_large_polygons(
            polygon_gdf_with_point, self.closing_fence_area_with_mast_threshold
        )
        polygon_gdf_without_point = selection.remove_large_polygons(
            polygon_gdf_without_point, self.closing_fence_area_threshold
        )

        # Combine filtered polygons
        faces_gdf = concat(
            [polygon_gdf_with_point, polygon_gdf_without_point], ignore_index=True
        )

        # Remove the surrounding fence lines of small closed areas with masts considered
        result_gdf = selection.remove_parts_of_lines_on_polygon_edges(
            result_gdf, faces_gdf
        )

        # Remove short fence lines
        result_gdf = selection.remove_disconnected_short_lines(
            result_gdf, self.fence_length_threshold
        )

        # Return simplified fence lines
        result_gdf.geometry = result_gdf.geometry.simplify(
            self.simplification_tolerance
        )

        return result_gdf
