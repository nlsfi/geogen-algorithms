#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import geopandas as gpd
import pandas as pd
import shapely.ops
import shapelysmooth
from pygeoops import centerline, remove_inner_rings, simplify  # noqa: SC200
from shapely import (
    LineString,
    Point,
    Polygon,
    affinity,
    force_2d,
    symmetric_difference,
)

from geogenalg.core.exceptions import GeometryOperationError

# TODO: don't use relative imports when moving this to a library
from ..core.geometry import (  # noqa: TID252
    LineExtendFrom,
    elongation,
    explode_line,
    extend_line_to_nearest,
    extract_interior_rings_gdf,
    lines_to_segments,
    move_to_point,
    oriented_envelope_dimensions,
    scale_line_to_length,
)


def generalize_watercourse_areas(  # noqa: C901, PLR0913, PLR0915
    areas: gpd.GeoDataFrame,
    *,
    max_width: float = 30,
    width_check_precision: float = 5,
    minimum_line_length: float = 200,
    minimum_polygon_area: float = 10000,
    min_island_area: float = 2500,
    min_island_elongation: float = 3.349,
    min_island_width: float = 185,
    exaggerate_island_by: float = 3,
) -> dict[str, gpd.GeoSeries]:
    """Generalize polygonal watercourse areas.

    Generalizes polygonal watercourse areas so that thin portions of the watercourse
    will be transformed into linestrings while wider areas will remain as polygons
    and simplified. Small islands will be removed, thin and long islands will be
    exaggerated.

    Returns:
        A dictionary containing a GeoDataFrame for both the new generated line
        features and the remaining modified polygon features.

    """
    # the basic steps of this algorithm are to
    # 1) create the center lines out of the watercourse areas
    # 2) divide those center lines into small segments
    # 3) check the width of the watercourse area at the centroid of each segment
    # 4) if the width at each segment is short enough keep those segments around
    # 5) recombine segments into linestrings
    # 6) filter out short linestrings
    # 7) create polygons out of the segments which follow the widths at each segment
    # 8) use the polygons to remove those portions out of the original watercourse areas
    # 9) filter out small remaining areas
    # 10) recalculate center lines for the areas determined to be turned into lines
    # 11) extend center lines to the remaining areas
    # 12) post-processing (simplify & smooth)

    # let's not modify original gdf
    watercourses = areas.copy()

    def exaggerate_thin_island(island: Polygon) -> Polygon:
        try:
            width = oriented_envelope_dimensions(island.oriented_envelope).width
            if elongation(island) >= min_island_elongation and width < min_island_width:
                return island.buffer(distance=exaggerate_island_by, quadsegs=2)  # noqa: SC200
        except GeometryOperationError:
            pass

        return island

    # dissolve and explode to combine any touching areas so that
    # a) islands which reside at the border of two split areas will be extracted
    # b) center lines will be correctly formed
    watercourses = watercourses.dissolve().explode(ignore_index=True)

    watercourses.geometry = watercourses.geometry.apply(
        lambda x: remove_inner_rings(x, 0, crs=watercourses.crs)
    )
    islands = extract_interior_rings_gdf(areas)
    islands = islands.loc[islands.geometry.area >= min_island_area]
    islands.geometry = islands.geometry.apply(exaggerate_thin_island)
    watercourses.geometry = watercourses.geometry.difference(islands.unary_union)

    watercourses_union = watercourses.union_all()

    # define inner functions which are specific to this algorithm
    # to be applied to various GeoSeries or used to loc[] rows

    def snap_to_watercourse(segment: LineString) -> LineString:
        extend: bool = True

        start_point = Point(segment.coords[0])
        end_point = Point(segment.coords[-1])

        if start_point.disjoint(watercourses_union) and end_point.disjoint(
            watercourses_union
        ):
            return shapely.intersection(segment, watercourses_union)

        extend_from = LineExtendFrom.BOTH

        if start_point.intersects(watercourses_union) and end_point.intersects(
            watercourses_union
        ):
            extend_from = LineExtendFrom.BOTH
        elif not start_point.intersects(watercourses_union) and end_point.intersects(
            watercourses_union
        ):
            extend_from = LineExtendFrom.END
        elif start_point.intersects(watercourses_union) and not end_point.intersects(
            watercourses_union
        ):
            extend_from = LineExtendFrom.START
        else:
            extend = False

        if extend:
            return shapely.intersection(
                extend_line_to_nearest(
                    segment,
                    watercourses_union.boundary,
                    extend_from,
                ),
                watercourses_union,
            )

        return segment

    def perpendicular_segment(
        original: LineString, centroid: Point, length: float | None = None
    ) -> LineString:
        segment = move_to_point(original, centroid)
        segment = affinity.rotate(segment, angle=90)

        return scale_line_to_length(
            segment, length if length is not None else original.length
        )

    def width_ruler(segment: LineString) -> LineString:
        return perpendicular_segment(segment, segment.centroid, max_width)

    def ruler_inside_watercourses(segments: gpd.GeoSeries) -> pd.Series:
        return pd.Series(
            [width_ruler(segment).within(watercourses_union) for segment in segments]
        )

    def watercourse_width_ruler(segment: LineString) -> LineString:
        segment = perpendicular_segment(segment, segment.centroid, max_width)
        return snap_to_watercourse(segment)

    def create_polygon_remover(input_line: LineString) -> Polygon:
        segmentized = input_line.segmentize(1)
        line_segments = explode_line(segmentized)

        buffers = [
            watercourse_width_ruler(segment).buffer(segment.length * 2)
            for segment in line_segments[2:-2]
        ]

        return shapely.ops.unary_union(buffers)

    center_lines = watercourses.copy()

    center_lines.geometry = (
        # center lines are still polygons here, segmentize to add vertices.
        # this results in a less jagged center line
        center_lines.geometry.segmentize(1)
        .apply(
            lambda geom: centerline(  # noqa: SC200
                geom,
                densify_distance=-1,
                min_branch_length=30,
                simplifytolerance=0,  # noqa: SC200
            )
        )
        .apply(lambda geom: simplify(geometry=geom, tolerance=10, algorithm="vw"))
        # now center lines are actual lines, segmentize again to allow more
        # precise checking of watercourse width
        .segmentize(width_check_precision)
    )

    segments = lines_to_segments(center_lines.geometry.explode(ignore_index=True))

    under_designated_width = gpd.GeoSeries(
        shapely.ops.linemerge(
            segments.loc[~ruler_inside_watercourses(segments.geometry)].unary_union
        ),
        crs=watercourses.crs,
    ).explode(ignore_index=True)

    under_designated_width = under_designated_width.loc[
        under_designated_width.length > minimum_line_length
    ]

    removers = under_designated_width.copy().apply(create_polygon_remover)

    remaining_areas = (
        watercourses.geometry.difference(removers.unary_union)
        .explode(ignore_index=True)
        .geometry
    )
    remaining_areas = remaining_areas.loc[
        remaining_areas.intersects(center_lines.unary_union)
    ]
    remaining_areas = remaining_areas.loc[remaining_areas.area >= minimum_polygon_area]

    # create center lines before simplifying areas for a better result
    remaining_center_lines = (
        gpd.GeoSeries(
            data=symmetric_difference(watercourses_union, remaining_areas.unary_union),
            crs=remaining_areas.crs,
        )
        .segmentize(1)
        .apply(
            lambda geom: centerline(  # noqa: SC200
                geom,
                densify_distance=-1,
                min_branch_length=30,
                simplifytolerance=0,  # noqa: SC200
            )
        )
        .explode(ignore_index=True)
    )

    # simplify areas first so center lines snap to the simplified version
    remaining_areas = remaining_areas.apply(
        lambda geom: simplify(geometry=geom, tolerance=2)
    ).apply(
        lambda geom: shapely.make_valid(shapelysmooth.taubin_smooth(force_2d(geom)))  # noqa: SC200
    )

    remaining_center_lines = remaining_center_lines.apply(
        lambda geom: extend_line_to_nearest(
            line=geom,
            extend_to=remaining_areas.geometry.unary_union.boundary,
            extend_from=LineExtendFrom.BOTH,
            tolerance=15,
        ),
    ).apply(lambda geom: simplify(geometry=geom, tolerance=10, algorithm="vw"))

    return {
        "areas": remaining_areas,
        "lines": remaining_center_lines,
    }
