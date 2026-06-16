#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from typing import ClassVar, override

from geopandas import GeoDataFrame, GeoSeries
from pydantic import Field
from shapely import MultiPoint, Polygon, force_2d

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.continuity import get_segments_in_polygon_exteriors_but_not_in_lines
from geogenalg.core.exceptions import GeometryOperationError
from geogenalg.core.geometry import (
    assign_nearest_z,
    chaikin_smooth_keep_topology,
    extract_interior_rings,
    largest_part,
    perforate_polygon_with_gdf_exteriors,
)
from geogenalg.exaggeration import (
    exaggerate_thin_polygons,
    extract_narrow_polygon_parts,
)


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

    @staticmethod
    def _get_skip_coords(data: GeoDataFrame, shoreline: GeoDataFrame) -> MultiPoint:
        segments = get_segments_in_polygon_exteriors_but_not_in_lines(data, shoreline)

        if segments.empty:
            return MultiPoint()

        points = segments.extract_unique_points().union_all()

        if not isinstance(points, MultiPoint):
            msg = "Result is not a MultiPoint."
            raise GeometryOperationError(msg)

        return force_2d(points)

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        if self.reference_key in reference_data:
            shoreline_gdf = reference_data[self.reference_key]
            # Extract out any points which are found in the input (Polygon)
            # data, but not the shoreline LineString data. This allows to
            # determine non-shoreline vertices (e.g. territorial water borders
            # in case of sea areas) and skip smoothing them later.
            skip_coords = GeneralizeWaterAreas._get_skip_coords(data, shoreline_gdf)
        else:
            skip_coords = MultiPoint()

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

            # don't exaggerate near common feature edges
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

        gdf.geometry = gdf.geometry.simplify_coverage(
            self.area_simplification_tolerance,
        )
        islands.geometry = islands.geometry.simplify(
            self.island_simplification_tolerance,
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

        return assign_nearest_z(data, gdf)
