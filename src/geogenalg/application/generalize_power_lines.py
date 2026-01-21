#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import cast, override

from geopandas import GeoDataFrame
from pandas import Series, concat
from shapely import LineString, MultiLineString, MultiPolygon, Polygon, concave_hull, delaunay_triangles, is_empty, polygonize, shortest_line, simplify, unary_union
from shapely.affinity import rotate
from shapely.ops import linemerge
from pygeoops import centerline

from geogenalg.analyze import flag_parallel_lines, get_parallel_line_areas
from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.core.geometry import explode_line, scale_line_to_length, segment_direction
from geogenalg.utility.validation import check_gdf_geometry_type


@dataclass(frozen=True)
class GeneralizePowerLines(BaseAlgorithm):
    """Generalize lines representing power lines.

    The algorithm does the following steps:
    - Merges parallel power lines
    - Ensures network connectivity with power stations
    - Removes short lines
    - Simplifies lines
    """

    distance_threshold_for_parallel_lines: float = 25.0
    """Minimum distance for lines to be considered to be parallel."""
    classes_for_merge_parallel_lines: list[int | str] = field(default_factory=list)
    """Attribute values to distinguish which types of parallel lines will be merged."""
    power_line_class_column: str = "kohdeluokka"
    """Column in which attribute values to distinguish power line types reside."""
    power_line_length_threshold: float = 100.0
    """Minimum length of a power line."""
    simplification_tolerance: float = 0.0
    """Tolerance used for geometry simplification."""

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        if not check_gdf_geometry_type(data, ["LineString"]):
            msg = "Input data must only contain LineStrings."
            raise GeometryTypeError(msg)

        gdf = cast("GeoDataFrame", data.copy())

        gdf = cast("GeoDataFrame", gdf.loc[gdf[self.power_line_class_column] == 22311])

        parallel_areas = get_parallel_line_areas(
            gdf,
            self.distance_threshold_for_parallel_lines,
            allowed_direction_difference=10,
            segmentize_distance=50,
        )

        # parallel_areas.geometry = parallel_areas.geometry.segmentize(5)
        # parallel_areas.geometry = parallel_areas.geometry.apply(
        #     lambda geom: centerline(
        #         geom,
        #         simplifytolerance=0.5,
        #         min_branch_length=10000000000000000000,
        #     ),
        # )

        return parallel_areas
