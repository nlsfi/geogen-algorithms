#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from typing import ClassVar, cast, override

from geopandas import GeoDataFrame, GeoSeries
from pydantic import Field
from pygeoops import simplify
from shapely import MultiPoint, Polygon, force_2d, shortest_line, union_all
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.attributes import inherit_attributes_for_lines_by_buffer
from geogenalg.continuity import (
    get_segments_in_polygon_boundary_but_not_in_lines,
)
from geogenalg.core.exceptions import GeometryOperationError, MissingReferenceError
from geogenalg.core.geometry import (
    LineExtendFrom,
    assign_nearest_z,
    chaikin_smooth_keep_topology,
    ensure_geoms,
    extend_line_by,
    extract_interior_rings,
    largest_part,
    perforate_polygon_with_gdf_exteriors,
    split_linear_geometry,
)
from geogenalg.exaggeration import (
    exaggerate_thin_polygons,
    extract_narrow_polygon_parts,
)
from geogenalg.identity import hash_duplicate_indexes
from geogenalg.utility.dataframe_processing import combine_gdfs


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

    A shoreline reference data may optionally be entered. This affects the
    algorithm so that any vertices not present in the shoreline data will not
    be modified while smoothing. This makes sense to use with sea part
    features, so that territorial water borders are not modified.
    """

    min_area: float = Field(4000.0, gt=0)
    """Features under this area will be removed."""
    area_simplification_tolerance: float = Field(10.0, gt=0)
    """Simplification tolerance used for water areas."""
    thin_section_width: float = Field(20.0, gt=0)
    """Sections under this width will be exaggerated."""
    thin_section_min_size: float = Field(200.0, gt=0)
    """Don't exaggerate thin sections under this size."""
    thin_section_exaggerate_by: float = Field(3.0, gt=0)
    """By how many CRS units thin sections will be exaggerated."""
    island_min_area: float = Field(100.0, gt=0)
    """Islands under this area will be removed."""
    island_min_width: float = Field(185.0, gt=0)
    """Islands under this width will be considered for exaggeration."""
    island_min_elongation: float = Field(0.25, ge=0.0, le=1.0)
    """Islands under this elongation will be considered for exaggeration."""
    island_exaggerate_by: float = Field(3.0, gt=0)
    """By how many CRS units thin islands will be exaggerated."""
    island_simplification_tolerance: float = Field(10.0, gt=0)
    """Simplification tolerance used for islands."""
    smoothing_passes: int = Field(3, ge=0)
    """How many smoothing passes will be performed. Each smoothing passes
    (nearly) doubles the vertex count."""
    include_new_shoreline: bool = False
    """If true, new shoreline will be created and included in the output.
    If set true, the reference shoreline data is mandatory."""
    preserve_shoreline_sections_column: str | None = None
    """Name of column used to select shoreline features whose vertices
    are preserved as is."""
    preserve_shoreline_sections_values: frozenset[int | str] = frozenset()
    """Types of shoreline whose vertices are preserved as is."""
    reference_key: str = "shoreline"
    """Reference data key for shoreline data. This optional reference data is
    intended to be a linestring dataset which (mostly) follows the input data
    exterior and shares its vertices. It is used to identify segments in the
    water area polygons which are not present in the shoreline, f.e. shared
    segments between water area features, or territorial sea borders. These
    segments are prevented from being smoothed. The shoreline data has to be of
    the previous scale and has to match the input data."""

    valid_input_geometry_types: ClassVar = {"Polygon"}
    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=False,
            valid_geometry_types={
                "LineString",
            },
        ),
    }

    def _get_segments_and_skip_coords(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> tuple[GeoDataFrame, MultiPoint]:
        if self.reference_key in reference_data:
            shoreline_gdf = reference_data[self.reference_key]

            # Extract out any points which are found in the input (Polygon)
            # data, but not the shoreline LineString data. This allows to
            # determine non-shoreline vertices (e.g. territorial water borders
            # in case of sea areas) and skip smoothing them later.
            segments = get_segments_in_polygon_boundary_but_not_in_lines(
                data,
                shoreline_gdf,
            )

            skip_coords = (
                MultiPoint()
                if segments.empty
                else segments.extract_unique_points().union_all()
            )
            if not isinstance(skip_coords, MultiPoint):
                msg = "Result is not a MultiPoint."
                raise GeometryOperationError(msg)

            # If specified, select shoreline features to preserve (no
            # simplification or smoothing) and add to skip_coords
            if (
                self.preserve_shoreline_sections_column is not None
                and self.preserve_shoreline_sections_values
            ):
                additional_skip_coords = (
                    shoreline_gdf.loc[
                        shoreline_gdf[self.preserve_shoreline_sections_column].isin(
                            self.preserve_shoreline_sections_values
                        )
                    ]
                    .geometry.extract_unique_points()
                    .union_all()
                )
                skip_coords = union_all([skip_coords, additional_skip_coords])

            skip_coords = force_2d(skip_coords)
        else:
            if self.include_new_shoreline:
                raise MissingReferenceError
            skip_coords = MultiPoint()
            segments = GeoDataFrame(geometry=[], crs=data.crs)

        return segments, skip_coords

    @staticmethod
    def _get_shoreline_splitters(
        unmodified_shoreline: GeoDataFrame,
        new_unsplit_shoreline: BaseGeometry,
        *,
        max_distance: float = 20,
    ) -> MultiLineString:
        # For the best shoreline splitting result, build splitter lines which
        # cut through the lines. Splitting by points (even if snapped) can be
        # inconsistent.
        def _to_splitter(point: Point | MultiPoint) -> MultiLineString:
            lines = []

            for p in ensure_geoms(point):
                line = shortest_line(
                    p,
                    new_unsplit_shoreline,
                )

                if line.length > max_distance:
                    continue

                if line.length == 0:
                    # This means the point is directly on a vertex. In order
                    # for the splitting to function we need the splitter
                    # geometries to be lines which cut through the
                    # line-to-split. Therefore create a tiny line around the
                    # vertex.
                    lines.append(
                        LineString(
                            [
                                [p.x, p.y + 0.00001],
                                [p.x, p.y],
                                [p.x, p.y - 0.00001],
                            ]
                        ),
                    )
                    continue

                lines.append(
                    extend_line_by(
                        line,
                        0.00001,
                        LineExtendFrom.END,
                    )
                )
            return MultiLineString(lines)

        splitters = unmodified_shoreline.copy()
        splitters.geometry = splitters.boundary
        splitters = splitters.loc[~splitters.geometry.is_empty]
        splitters.geometry = splitters.geometry.apply(_to_splitter)

        result = splitters.union_all()

        if isinstance(result, LineString):
            return MultiLineString([result])

        if result.is_empty:
            return MultiLineString()

        if not isinstance(result, MultiLineString):
            msg = "Did not get MultiLineString"
            raise GeometryOperationError(msg)

        return result

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        segments, skip_coords = self._get_segments_and_skip_coords(data, reference_data)

        gdf = data.copy()

        if self.thin_section_exaggerate_by != 0.0:
            thin_sections = (
                extract_narrow_polygon_parts(gdf, self.thin_section_width)
                .explode(index_parts=True)
                .geometry
            )

            thin_sections = thin_sections.loc[
                thin_sections.geometry.area > self.thin_section_min_size
            ].buffer(self.thin_section_exaggerate_by)

            # Skip exaggerating near coordinates which should be preserved
            thin_sections = thin_sections.loc[
                thin_sections.geometry.disjoint(skip_coords)
            ]
            # TODO: fix this issue by removing the generated overlap?
            # A straightforward intersection removal would move the edge however.

            def add_exaggerated_parts(geom: Polygon) -> Polygon:
                intersecting_geoms = thin_sections.loc[
                    thin_sections.geometry.intersects(geom)
                ].union_all()
                return geom.union(intersecting_geoms)

            gdf.geometry = gdf.geometry.apply(add_exaggerated_parts)

        # Extract from unary union to catch islands which are between lake parts
        islands_geom = extract_interior_rings(gdf.union_all())
        islands = GeoDataFrame(
            geometry=GeoSeries(islands_geom).explode(index_parts=True)
        )

        gdf.geometry = gdf.geometry.apply(lambda geom: Polygon(geom.exterior))

        # When islands are extracted, we get just an exterior ring. Because of
        # recursive lakes and islands, it is required to add any lakes as
        # interior rings to the extracted islands.
        islands.geometry = islands.geometry.apply(
            lambda geom: perforate_polygon_with_gdf_exteriors(geom, gdf)
        )

        if self.island_exaggerate_by != 0.0:
            islands = exaggerate_thin_polygons(
                islands,
                self.island_min_width,
                self.island_min_elongation,
                self.island_exaggerate_by,
            )

        gdf.geometry = gdf.geometry.force_2d().apply(
            simplify,
            tolerance=self.area_simplification_tolerance,
            algorithm="vw",
            preserve_topology=True,
            preserve_common_boundaries=True,
            keep_points_on=skip_coords,
        )
        islands.geometry = islands.geometry.force_2d().apply(
            simplify,
            tolerance=self.island_simplification_tolerance,
            algorithm="vw",
            preserve_topology=True,
            preserve_common_boundaries=True,
            keep_points_on=skip_coords,
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
        gdf.geometry = gdf.geometry.apply(largest_part)

        gdf.geometry = chaikin_smooth_keep_topology(
            gdf.geometry,
            iterations=self.smoothing_passes,
            extra_skip_coords=skip_coords,
        )

        if self.include_new_shoreline and not gdf.empty:
            shoreline = cast("GeoDataFrame", gdf.copy())
            shoreline.geometry = shoreline.geometry.boundary

            # Remove segments which were originally identified to be present in
            # the unmodified water area boundaries but not in unmodified
            # shoreline. More simply, this removes segments which are not part
            # of an actual shoreline.
            shoreline = shoreline.overlay(
                GeoDataFrame(geometry=[segments.union_all()], crs=data.crs),
                how="difference",
            ).explode()

            splitters = GeneralizeWaterAreas._get_shoreline_splitters(
                reference_data[self.reference_key],
                shoreline.union_all(),
            )

            shoreline.geometry = shoreline.geometry.apply(
                lambda geom: split_linear_geometry(geom, splitters)
            )
            shoreline = shoreline.explode().reset_index(drop=True)

            shoreline = inherit_attributes_for_lines_by_buffer(
                reference_data[self.reference_key],
                shoreline,
                buffer_distance=max(
                    self.island_exaggerate_by,
                    self.thin_section_exaggerate_by,
                )
                + 1,
            )

            gdf = combine_gdfs([gdf, shoreline])

        gdf = hash_duplicate_indexes(gdf, "water_areas")
        return assign_nearest_z(data, gdf)
