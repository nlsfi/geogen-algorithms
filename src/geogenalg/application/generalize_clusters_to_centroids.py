#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain
from typing import Any, ClassVar

from geopandas import GeoDataFrame
from pandas import Series, concat

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.cluster import get_cluster_centroids


@supports_identity
@dataclass(frozen=True)
class GeneralizePointClustersAndPolygonsToCentroids(BaseAlgorithm):
    """Reduces polygons and point clusters to single points.

    Input can be point and/or polygons for which centroids are possibly generated.

    Reference data can be polygons, within which the centroids will be placed.

    Output is generalized data, which contains the newly
    created centroids and all input features which were not turned into
    centroids. These can be distinguished by the value in the column of
    the name set by the feature_type_column attribute.
    """

    cluster_distance: float = 30.0
    """Points within this distance of each other will be clustered."""
    polygon_min_area: float = 4000.0
    """Polygons with area smaller than this will be turned into a point."""
    feature_type_column: str = "feature_type"
    """Name of column containing type of output feature."""
    aggregation_functions: dict[str, Callable[[Series], Any] | str] | None = None
    """Dictionary containing keys corresponding to a column in a GeoDataFrame
    and a function which will aggregate the column's values when creating
    centroid from multiple points. If the function is given as a string, it
    must correspond to Pandas's aggregation function names. If no function is
    given "first" will be used by default."""

    valid_input_geometry_types: ClassVar = {"Point", "Polygon"}

    def _process_polygons(
        self,
        polygons: GeoDataFrame,
    ) -> GeoDataFrame:
        gdf = polygons.loc[polygons.geometry.area < self.polygon_min_area].copy()
        gdf.geometry = gdf.geometry.apply(lambda geom: geom.point_on_surface())

        return gdf

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],  # noqa: ARG002
    ) -> GeoDataFrame:
        index_name = data.index.name

        points, polygons = (
            data.loc[data.geometry.type == "Point"],
            data.loc[data.geometry.type == "Polygon"],
        )

        clusters_from_points = GeoDataFrame()
        if not points.empty:
            clusters_from_points = get_cluster_centroids(
                points,
                self.cluster_distance,
                aggregation_functions=self.aggregation_functions,
            )

            clusters_from_points = clusters_from_points.set_index(
                clusters_from_points["old_ids"].apply(
                    lambda ids: sha256(
                        b"pointcluster" + "".join(sorted(ids)).encode()
                    ).hexdigest(),
                ),
            )
            clusters_from_points[self.feature_type_column] = "centroid_from_point"

        clusters_from_polygons = GeoDataFrame()
        if not polygons.empty:
            clusters_from_polygons = self._process_polygons(polygons)
            clusters_from_polygons[self.feature_type_column] = "centroid_from_polygon"

        if not clusters_from_points.empty:
            ids_to_remove = list(chain.from_iterable(clusters_from_points["old_ids"]))

            points = points.loc[~points.index.isin(ids_to_remove)]

        if not clusters_from_polygons.empty:
            polygons = polygons.loc[(polygons.geometry.area >= self.polygon_min_area)]

        if "old_ids" in clusters_from_points.columns:
            clusters_from_points = clusters_from_points.drop(
                columns=["old_ids"],
                axis=1,
            )

        points[self.feature_type_column] = "unchanged_point"
        polygons[self.feature_type_column] = "unchanged_polygon"

        result = GeoDataFrame(
            concat([clusters_from_points, clusters_from_polygons, points, polygons]),
        )

        result.index.name = index_name

        return result
