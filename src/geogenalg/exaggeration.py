#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from dataclasses import dataclass
from typing import Literal

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pygeoops import PrimitiveType, collection_extract
from shapely import (
    BufferCapStyle,
    BufferJoinStyle,
    Polygon,
    buffer,
    get_coordinates,
    make_valid,
    oriented_envelope,
)
from shapely.geometry.base import BaseGeometry

from geogenalg.core.exceptions import GeometryTypeError
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
    *,
    sliver_cleaning_distance: float = 0.1,
) -> GeoDataFrame:
    """Extract polygon parts narrower than the threshold.

    Args:
    ----
        input_gdf: GeoDataFrame containing Polygon or MultiPolygon geometries
        threshold: Minimum width for a polygon part. Parts narrower than this are
              extracted.
        sliver_cleaning_distance: Slivers narrower than twice this distance are
            removed from the result with a negative and positive buffer. Set to
            0 to skip cleaning, which is faster but the result may then contain
            slivers produced by the difference operation.

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

    if sliver_cleaning_distance < 0:
        msg = "Sliver cleaning distance must not be negative."
        raise ValueError(msg)

    wide_parts = (
        in_parallel(input_gdf.geometry)
        .buffer(-(threshold / 2), cap_style="flat", join_style="mitre")
        .buffer(threshold / 2, cap_style="flat", join_style="mitre")
        .resolve_geoseries()
    )
    wide_parts = make_valid(wide_parts)
    wide_parts = GeoSeries(
        collection_extract(wide_parts, PrimitiveType.POLYGON),
        crs=input_gdf.crs,
    )

    gdf = input_gdf.copy()
    narrow_parts = in_parallel(gdf.geometry).difference(wide_parts)

    if sliver_cleaning_distance > 0:
        # TODO: consider minimum_width() for this?
        narrow_parts = narrow_parts.buffer(
            -sliver_cleaning_distance, cap_style="flat", join_style="mitre"
        ).buffer(sliver_cleaning_distance, cap_style="flat", join_style="mitre")

    gdf.geometry = narrow_parts.resolve_geoseries()

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

    gdf = input_gdf.copy()
    geoms = gdf.geometry.to_numpy()

    corners = get_coordinates(oriented_envelope(geoms)).reshape(len(geoms), 5, 2)

    side_1 = np.hypot(*(corners[:, 1] - corners[:, 0]).T)
    side_2 = np.hypot(*(corners[:, 2] - corners[:, 1]).T)
    widths = np.minimum(side_1, side_2)
    heights = np.maximum(side_1, side_2)

    is_thin = (widths <= elongation_threshold * heights) & (widths <= width_threshold)
    geoms[is_thin] = buffer(  # ty: ignore[no-matching-overload]
        geoms[is_thin],
        exaggerate_by,
        quad_segs=buffer_options.quad_segments,  # noqa: SC200
        # There's a bug in the shapely stub typing this as BufferJoinStyle,
        # hence the ignore.
        cap_style=buffer_options.cap_style,
        join_style=buffer_options.join_style,
        mitre_limit=buffer_options.mitre_limit,
        single_sided=buffer_options.single_sided,
    )
    gdf.geometry = geoms

    return gdf
