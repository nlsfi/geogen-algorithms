#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import operator
from dataclasses import dataclass
from typing import override

from geopandas import GeoDataFrame, GeoSeries
from shapely import MultiPoint, MultiPolygon, Point, Polygon, force_2d

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.core.geometry import (
    assign_nearest_z,
    chaikin_smooth_keep_topology,
    extract_interior_rings,
    perforate_polygon_with_gdf_exteriors,
)
from geogenalg.exaggeration import (
    exaggerate_thin_polygons,
    extract_narrow_polygon_parts,
)
from geogenalg.utility.validation import check_gdf_geometry_type


@supports_identity
@dataclass(frozen=True)
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

    min_area: float = 4000.0
    """Features under this area will be removed."""
    area_simplification_tolerance: float = 10.0
    """Simplification tolerance used for water areas."""
    thin_section_width: float = 20.0
    """Sections under this width will be exaggerated."""
    thin_section_min_size: float = 200.0
    """Don't exaggerate thin sections under this size."""
    thin_section_exaggerate_by: float = 3.0
    """By how many CRS units thin sections will be exaggerated."""
    island_min_area: float = 100.0
    """Islands under this area will be removed."""
    island_min_width: float = 185.0
    """Islands under this width will be considered for exaggeration."""
    island_min_elongation: float = 0.25
    """Islands under this elongation will be considered for exaggeration."""
    island_exaggerate_by: float = 3.0
    """By how many CRS units thin islands will be exaggerated."""
    island_simplification_tolerance: float = 10.0
    """Simplification tolerance used for islands."""
    smoothing_passes: int = 3
    """How many smoothing passes will be performed. Each smoothing passes
    (nearly) doubles the vertex count."""
    reference_key: str = "shoreline"
    """Reference data key to use as a shoreline layer for preventing smoothing of
    non-shoreline vertices."""

    @staticmethod
    def _get_skip_coords(data: GeoDataFrame, shoreline: GeoDataFrame) -> MultiPoint:
        data_points = data.extract_unique_points().union_all()
        shoreline_points = shoreline.extract_unique_points().union_all()
        skip_coords = force_2d(data_points.difference(shoreline_points))

        if isinstance(skip_coords, Point) and not skip_coords.is_empty:
            skip_coords = MultiPoint([skip_coords])

        if skip_coords.is_empty:
            skip_coords = MultiPoint()

        if not isinstance(skip_coords, MultiPoint):
            skip_coords = MultiPoint()

        return skip_coords

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        if not check_gdf_geometry_type(data, ["Polygon"]):
            msg = "Input data must only contain Polygons."
            raise GeometryTypeError(msg)

        if not 0.0 <= self.island_min_elongation <= 1.0:
            msg = "Minimum island elongation must be between 0.0 and 1.0."
            raise ValueError(msg)

        if self.reference_key in reference_data:
            shoreline_gdf = reference_data[self.reference_key]
            if not check_gdf_geometry_type(shoreline_gdf, ["LineString"]):
                msg = "Reference data must only contain LineStrings."
                raise GeometryTypeError(msg)

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
        def multipolygon_to_single_keep_largest(
            geom: MultiPolygon | Polygon,
        ) -> Polygon:
            if isinstance(geom, Polygon):
                return geom

            areas = [(geom.area, geom) for geom in geom.geoms]
            areas.sort(key=operator.itemgetter(0), reverse=True)

            return areas[0][1]

        mask = gdf.geometry.geom_type == "MultiPolygon"
        gdf.loc[mask, gdf.geometry.name] = gdf.loc[mask].geometry.apply(
            multipolygon_to_single_keep_largest
        )

        gdf.geometry = chaikin_smooth_keep_topology(
            gdf.geometry,
            extra_skip_coords=skip_coords,
        )

        return assign_nearest_z(data, gdf)
