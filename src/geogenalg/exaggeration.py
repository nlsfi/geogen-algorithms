#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from dataclasses import dataclass
from typing import Literal

from geopandas import GeoDataFrame
from shapely import (
    BufferCapStyle,
    BufferJoinStyle,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.core.geometry import elongation, oriented_envelope_dimensions
from geogenalg.core.parallel import in_parallel
from geogenalg.utility.validation import check_gdf_geometry_type


@dataclass
class BufferOptions:
    """Data class for passing options to buffer function."""

    quad_segments: int = 16
    cap_style: BufferCapStyle | Literal["round", "square", "flat"] = "round"
    join_style: BufferJoinStyle | Literal["round", "mitre", "bevel"] = "round"
    mitre_limit: float = 5
    single_sided: bool = False


def buffer_with_options(
    geom: BaseGeometry,
    distance: float,
    options: BufferOptions,
) -> Polygon:
    """Buffer geometry with options from BufferOptions object.

    Returns
    -------
        Buffered geometry.

    """
    return geom.buffer(
        distance=distance,
        quad_segs=options.quad_segments,  # noqa: SC200
        cap_style=options.cap_style,
        join_style=options.join_style,
        mitre_limit=options.mitre_limit,
        single_sided=options.single_sided,
    )


def extract_narrow_polygon_parts(
    input_gdf: GeoDataFrame,
    threshold: float,
) -> GeoDataFrame:
    """Extract polygon parts narrower than the threshold.

    Args:
    ----
        input_gdf: GeoDataFrame containing Polygon or MultiPolygon geometries
        threshold: Minimum width for a polygon part. Parts narrower than this are
              extracted.

    Returns:
    -------
        GeoDataFrame containing the narrow polygon parts. **Each row corresponds
        to a row in the input GeoDataFrame.** If an input polygon does not
        contain any parts narrower than the threshold, the corresponding
        geometry in the result will be empty (a GeometryCollection), but the row
        will still be present.

    """
    if not check_gdf_geometry_type(input_gdf, {"Polygon", "MultiPolygon"}):
        msg = "Extract narrow parts only supports Polygon or MultiPolygon geometries."
        raise GeometryTypeError(msg)

    wide_parts = (
        in_parallel(input_gdf.geometry)
        .buffer(-(threshold / 2), cap_style="flat", join_style="mitre")
        .buffer(threshold / 2, cap_style="flat", join_style="mitre")
        .resolve_geoseries()
    )

    gdf = input_gdf.copy()
    gdf.geometry = (
        in_parallel(gdf.geometry)
        .difference(wide_parts)
        # Clean small slivers
        # TODO: consider minimum_width() for this?
        .buffer(-0.1, cap_style="flat", join_style="mitre")
        .buffer(0.1, cap_style="flat", join_style="mitre")
        .resolve_geoseries()
    )

    return gdf


def exaggerate_thin_polygons(
    input_gdf: GeoDataFrame,
    width_threshold: float,
    elongation_threshold: float,
    exaggerate_by: float,
    buffer_options: BufferOptions | None = None,
) -> GeoDataFrame:
    """Exaggerate polygon if it's considered thin.

    If the polygon's oriented bounding box is under the width and elongation
    threshold, it will be considered thin.

    Args:
    ----
        input_gdf: GeoDataFrame containing polygon geometries.
        width_threshold: maximum value for width for polygon to be considered thin
        elongation_threshold: maximum value for elongation for polygon to be
            considered thin
        exaggerate_by: buffer distance for exaggeration
        buffer_options: options to pass to buffer function

    Returns:
    -------
        GeoDataFrame containing the input features with the thin polygons exaggerated.

    """
    if buffer_options is None:
        buffer_options = BufferOptions()

    modified = input_gdf.copy()

    def _process_polygon(geom: Polygon) -> Polygon:
        width = oriented_envelope_dimensions(geom).width

        if elongation(geom) <= elongation_threshold and width <= width_threshold:
            return buffer_with_options(geom, exaggerate_by, buffer_options)

        return geom

    modified.geometry = modified.geometry.apply(_process_polygon)  # ty: ignore[no-matching-overload]

    return modified
