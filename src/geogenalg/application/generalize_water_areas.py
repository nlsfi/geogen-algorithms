#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from itertools import pairwise
from typing import ClassVar, cast, override

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pandas.api.types import is_string_dtype
from pydantic import Field
from shapely import (
    MultiPoint,
    MultiPolygon,
    Polygon,
    STRtree,
    extract_unique_points,
    force_2d,
    get_coordinates,
    get_exterior_ring,
    get_parts,
    get_segments,  # ty: ignore[unresolved-import] (type stubs are for shapely 2.1)
    is_empty,
    length,
    linestrings,
    polygons,
    shortest_line,
    union,
    union_all,
)
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from geogenalg.application import (
    BaseAlgorithm,
    supports_identity,
)
from geogenalg.attributes import (
    inherit_attributes_for_lines_by_centroid_distances,
)
from geogenalg.continuity import (
    get_segments_in_polygon_boundary_but_not_in_lines,
)
from geogenalg.core.exceptions import GeometryOperationError
from geogenalg.core.geometry import (
    assign_nearest_z,
    chaikin_smooth_keep_topology,
    create_crossing_line_at_vertex,
    extract_interior_rings,
    get_topological_points,
    largest_part,
    perforate_polygons_with_gdf_exteriors,
    split_linear_geometry,
    union_by_predicate,
    visvalingam_whyatt_simplify_conditional,
)
from geogenalg.exaggeration import (
    exaggerate_thin_polygons,
    extract_narrow_polygon_parts,
)
from geogenalg.identity import hash_duplicate_indexes
from geogenalg.utility.dataframe_processing import combine_gdfs, copy_gdf_as_empty


def _disjoint_from(geoms: GeoSeries, other: BaseGeometry) -> np.ndarray:
    mask = np.ones(len(geoms), dtype=bool)
    if not other.is_empty:
        mask[geoms.sindex.query(other, predicate="intersects")] = False
    return mask


@supports_identity
class GeneralizeWaterAreas(BaseAlgorithm):
    """Generalizes polygonal water areas.

    The algorithm does the following:
    - Removes interior rings (islands) under given size
    - Removes water areas under given size
    - Exaggerates (buffer) thin sections of areas
    - Exaggerates thin islands
    - Simplifies the areas
    - Smooths the simplified areas, while retaining topology between areas

    Optionally linestring features may be entered in the input data. These are
    meant to represent shoreline features which (mostly) follows the exteriors
    of the polygonal water areas. These features are used to identify vertices
    which are not present in the shoreline f.e. shared segments between area
    features or territorial sea borders. These segments are prevented from
    being smoothed or simplified. Additionally you may define an attribute
    which selects specific shoreline features which will be not smoothed or
    simplified, which may be useful for preserving built up dock areas etc.

    If shoreline features are included in the input data, new shoreline
    features will be created from the generalized areas and included in the
    output.

    """

    min_area: float = Field(4000.0, ge=0)
    """Features under this area will be removed."""
    area_simplification_tolerance: float = Field(100.0, ge=0)
    """Simplification tolerance used for water areas."""
    thin_section_width: float = Field(20.0, ge=0)
    """Sections under this width will be exaggerated."""
    thin_section_min_size: float = Field(200.0, ge=0)
    """Don't exaggerate thin sections under this size."""
    thin_section_exaggerate_by: float = Field(3.0, ge=0)
    """By how many CRS units thin sections will be exaggerated."""
    island_min_area: float = Field(100.0, ge=0)
    """Islands under this area will be removed."""
    island_min_width: float = Field(185.0, ge=0)
    """Islands under this width will be considered for exaggeration."""
    island_min_elongation: float = Field(0.25, ge=0.0, le=1.0)
    """Islands under this elongation will be considered for exaggeration."""
    island_exaggerate_by: float = Field(3.0, ge=0)
    """By how many CRS units thin islands will be exaggerated."""
    island_simplification_tolerance: float = Field(50.0, ge=0)
    """Simplification tolerance used for islands."""
    smoothing_passes: int = Field(3, ge=0)
    """How many smoothing passes will be performed. Each smoothing passes
    (nearly) doubles the vertex count."""
    smoothing_maximum_cut_distance: float | None = None
    """If the distance between an original vertex and its potential smoothed
    vertex exceeds this distance the vertex will remain unchanged."""
    preserve_shoreline_sections_column: str | None = None
    """Name of column used to select shoreline features whose vertices are
    preserved as is. This affects results only if shoreline features are
    present in input data and preserve_shoreline_sections_values are
    provided."""
    preserve_shoreline_sections_values: frozenset[int | str] = frozenset()
    """Types of shoreline whose vertices are preserved as is. This affects
    results only if shoreline features are present in input data and
    preserve_shoreline_sections_column are provided."""

    valid_input_geometry_types: ClassVar = {"Polygon", "LineString"}

    def _get_shoreline_gdf_segments_and_skip_coords(
        self,
        data: GeoDataFrame,
    ) -> tuple[GeoDataFrame, GeoDataFrame, MultiPoint]:
        shoreline_gdf = data.loc[data.geometry.geom_type == "LineString"].copy()
        shoreline_gdf.geometry = shoreline_gdf.geometry.force_2d()

        if not shoreline_gdf.empty:
            # Extract out any points which are found in the input (Polygon)
            # data, but not the shoreline LineString data. This allows to
            # determine non-shoreline vertices (e.g. territorial water borders
            # in case of sea areas) and skip smoothing them later.
            segments = get_segments_in_polygon_boundary_but_not_in_lines(
                data.geometry.force_2d(),
                shoreline_gdf,
            )

            skip_coords = (
                MultiPoint()
                if segments.empty
                else segments.extract_unique_points().union_all()
            )

            # If specified, select shoreline features to preserve (no
            # simplification or smoothing) and add to skip_coords
            if (
                self.preserve_shoreline_sections_column is not None
                and self.preserve_shoreline_sections_values
            ):
                additional_skip_coords = shoreline_gdf.loc[
                    shoreline_gdf[self.preserve_shoreline_sections_column].isin(
                        self.preserve_shoreline_sections_values
                    )
                ].geometry

                if not additional_skip_coords.empty:
                    skip_coords = extract_unique_points(
                        union_all(
                            [
                                skip_coords,
                                additional_skip_coords.union_all(),
                            ],
                        )
                    )

            if not isinstance(skip_coords, MultiPoint):
                msg = "Result is not a MultiPoint."
                raise GeometryOperationError(msg)

            skip_coords = force_2d(skip_coords)
        else:
            skip_coords = MultiPoint()
            segments = GeoDataFrame(geometry=[], crs=data.crs)

        return shoreline_gdf, segments, skip_coords

    @staticmethod
    def _get_shoreline_splitters(
        unmodified_shoreline: GeoDataFrame,
        new_unsplit_shoreline: GeoSeries,
        *,
        max_distance: float = 20,
    ) -> list[MultiLineString]:
        """Build splitters for each new shoreline feature.

        Returns
        -------
            A MultiLineString per feature of new_unsplit_shoreline, containing
                the splitters which touch that feature.

        """
        # For the best shoreline splitting result, build splitter lines which
        # cut through the lines. Splitting by points (even if snapped) can be
        # inconsistent.
        no_splitters = [MultiLineString() for _ in range(len(new_unsplit_shoreline))]

        shoreline_union = cast(
            "LineString | MultiLineString", new_unsplit_shoreline.union_all()
        )
        segments = np.asarray(
            get_segments(get_parts(shoreline_union)),
            dtype=object,
        )
        tree = STRtree(segments)

        points = get_parts(unmodified_shoreline.geometry.boundary.to_numpy())
        points = points[~is_empty(points)]

        if len(points) == 0 or len(segments) == 0:
            return no_splitters

        nearest = tree.nearest(points)
        lines = shortest_line(points, segments[nearest])
        lengths = length(lines)

        within_range = lengths <= max_distance
        extend_by = 0.00001

        # Extend lines slightly so they are guaranteed to cross the line
        extendable = within_range & (lengths > 0)
        start, end = get_coordinates(lines[extendable]).reshape(-1, 2, 2).swapaxes(0, 1)
        direction = end - start
        new_end = end + direction / np.hypot(*direction.T)[:, np.newaxis] * extend_by
        splitter_lines: list[LineString] = list(
            np.asarray(
                linestrings(np.stack([start, end, new_end], axis=1)),
                dtype=object,
            )
        )

        # Points which form a line with length 0.0 sit directly on the shoreline.
        # These require special handling.
        on_shoreline = points[within_range & (lengths == 0)]
        point_idx, segment_idx = tree.query(
            on_shoreline,
            predicate="dwithin",
            distance=1e-6,
        )
        bounds = np.searchsorted(point_idx, np.arange(len(on_shoreline) + 1))
        segments_per_point = [segment_idx[a:b] for a, b in pairwise(bounds)]
        for point, connected in zip(on_shoreline, segments_per_point, strict=True):
            crossing = create_crossing_line_at_vertex(
                MultiLineString(segments[connected].tolist()),
                point,
                length=extend_by,
            )

            if not crossing.is_empty:
                splitter_lines.append(crossing)

        if not splitter_lines:
            return no_splitters

        # Only give each feature the splitters which actually touch it.
        splitter_parts = get_parts(union_all(splitter_lines))
        feature_idx, part_idx = STRtree(splitter_parts).query(
            new_unsplit_shoreline.to_numpy(),
            predicate="intersects",
        )
        parts_per_feature = np.split(
            part_idx,
            np.searchsorted(feature_idx, np.arange(1, len(new_unsplit_shoreline))),
        )

        return [
            MultiLineString(splitter_parts[np.sort(parts)].tolist())
            for parts in parts_per_feature
        ]

    def _build_generalized_shoreline(
        self,
        gdf: GeoDataFrame,
        original_shoreline: GeoDataFrame,
        non_shoreline_segments: GeoDataFrame,
    ) -> GeoDataFrame:
        shoreline = gdf.copy()

        if shoreline.empty:
            return GeoDataFrame(geometry=[], crs=gdf.crs)

        shoreline.geometry = shoreline.geometry.boundary

        # Remove segments which were originally identified to be present in
        # the unmodified water area boundaries but not in unmodified
        # shoreline. More simply, this removes segments which are not part
        # of an actual shoreline.
        shoreline = shoreline.overlay(
            GeoDataFrame(geometry=[non_shoreline_segments.union_all()], crs=gdf.crs),
            how="difference",
        ).explode()

        splitters = GeneralizeWaterAreas._get_shoreline_splitters(
            original_shoreline,
            shoreline.geometry,
        )

        shoreline.geometry = [
            MultiLineString([geom])
            if splitter.is_empty
            else split_linear_geometry(geom, splitter)
            for geom, splitter in zip(shoreline.geometry, splitters, strict=True)
        ]
        shoreline = shoreline.explode().reset_index(drop=True)

        return inherit_attributes_for_lines_by_centroid_distances(
            original_shoreline,
            shoreline,
        )

    @staticmethod
    def _simplify(
        geoseries: GeoSeries,
        tolerance: float,
        skip_coords: MultiPoint,
        *,
        has_shoreline: bool,
    ) -> GeoSeries:
        geoseries = geoseries.force_2d()

        coords_to_keep = (
            skip_coords
            if has_shoreline
            else MultiPoint(get_topological_points(geoseries))
        )

        return visvalingam_whyatt_simplify_conditional(
            geoseries,
            tolerance,
            skip_coords=coords_to_keep,
        )

    def _post_process(
        self,
        gdf: GeoDataFrame,
        original_data: GeoDataFrame,
        shoreline_gdf: GeoDataFrame,
        non_shoreline_segments: GeoDataFrame,
        skip_coords: MultiPoint,
    ) -> GeoDataFrame:
        gdf.geometry = chaikin_smooth_keep_topology(
            gdf.geometry,
            iterations=self.smoothing_passes,
            extra_skip_coords=skip_coords,
            distance_threshold=self.smoothing_maximum_cut_distance,
        )

        if not shoreline_gdf.empty and not gdf.empty:
            shoreline = self._build_generalized_shoreline(
                gdf,
                shoreline_gdf,
                non_shoreline_segments,
            )

            if not is_string_dtype(shoreline.index.dtype):
                shoreline = shoreline.set_index(shoreline.index.astype("string"))

            gdf = combine_gdfs([gdf, shoreline])

        gdf = hash_duplicate_indexes(gdf, "water_areas")
        return assign_nearest_z(original_data, gdf)

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        shoreline_gdf, non_shoreline_segments, skip_coords = (
            self._get_shoreline_gdf_segments_and_skip_coords(
                data,
            )
        )

        gdf = data.loc[data.geometry.geom_type == "Polygon"].copy()
        non_shoreline_union = non_shoreline_segments.union_all()

        if self.thin_section_exaggerate_by != 0.0:
            thin_sections = (
                extract_narrow_polygon_parts(gdf, self.thin_section_width)
                .explode(index_parts=True)
                .geometry
            )

            thin_sections = thin_sections.loc[
                thin_sections.geometry.area > self.thin_section_min_size
            ].buffer(self.thin_section_exaggerate_by)

            # Skip exaggerating near sections which should be preserved
            thin_sections = thin_sections.loc[
                _disjoint_from(thin_sections.geometry, non_shoreline_union)
                & _disjoint_from(thin_sections.geometry, skip_coords)
            ]
            # TODO: fix this issue by removing the generated overlap?
            # A straightforward intersection removal would move the edge however.

            exaggerated_parts = union_by_predicate(
                gdf.geometry,
                thin_sections,
                "intersects",
            )

            gdf.geometry = union(gdf.geometry.to_numpy(), exaggerated_parts)

        # Extract from unary union to catch islands which are between lake parts
        islands_geom = extract_interior_rings(
            cast("Polygon | MultiPolygon", gdf.union_all())
        )
        islands = GeoDataFrame(
            geometry=GeoSeries(islands_geom).explode(index_parts=True)
        )

        gdf.geometry = np.asarray(
            polygons(get_exterior_ring(gdf.geometry.to_numpy())),
            dtype=object,
        )

        # When islands are extracted, we get just an exterior ring. Because of
        # recursive lakes and islands, it is required to add any lakes as
        # interior rings to the extracted islands.
        islands.geometry = perforate_polygons_with_gdf_exteriors(
            islands.geometry,
            gdf,
        )

        if self.island_exaggerate_by != 0.0:
            islands = exaggerate_thin_polygons(
                islands,
                self.island_min_width,
                self.island_min_elongation,
                self.island_exaggerate_by,
            )

        gdf.geometry = self._simplify(
            gdf.geometry,
            self.area_simplification_tolerance,
            skip_coords,
            has_shoreline=not shoreline_gdf.empty,
        )
        islands.geometry = self._simplify(
            islands.geometry,
            self.island_simplification_tolerance,
            skip_coords,
            has_shoreline=not shoreline_gdf.empty,
        )

        islands = islands.loc[islands.geometry.area > self.island_min_area]

        # Delete small areas
        gdf = gdf.loc[gdf.geometry.area > self.min_area]

        # Add islands back in
        gdf.geometry = gdf.geometry.difference(islands.geometry.union_all())

        # The difference between the islands and areas might result in tiny
        # slivers and therefore change the geometry to a MultiPolygon. Resolve
        # this by transforming any MultiPolygons to Polygons, retaining only
        # the largest part.
        gdf.geometry = gdf.geometry.apply(largest_part)  # ty: ignore[no-matching-overload]

        if gdf.empty:
            return copy_gdf_as_empty(gdf)

        return self._post_process(
            gdf,
            data,
            shoreline_gdf,
            non_shoreline_segments,
            skip_coords,
        )
