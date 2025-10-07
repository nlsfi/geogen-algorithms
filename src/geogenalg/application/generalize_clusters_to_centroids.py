#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain
from typing import Any

from geopandas import GeoDataFrame
from pandas import Series, concat
from shapely import Point, Polygon, box

from geogenalg.application import BaseAlgorithm
from geogenalg.cluster import get_cluster_centroids
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.validation import check_gdf_geometry_type


@dataclass(frozen=True)
class GeneralizePointClustersAndPolygonsToCentroids(BaseAlgorithm):
    """Reduces polygons and point clusters to single points."""

    """Points within this distance of each other will be clustered."""
    cluster_distance: float
    """Polygons with area smaller than this will be turned into a point."""
    polygon_min_area: float
    """Name of column containing a unique identifier of input features."""
    unique_id_column: str
    """Name of column containing type of output feature."""
    feature_type_column: str
    """Dictionary containing keys corresponding to a column in a GeoDataFrame
    and a function which will aggregate the column's values when creating
    centroid from multiple points. If the function is given as a string, it
    must correspond to Pandas's aggregation function names. If no function is
    given "first" will be used by default."""
    aggregation_functions: dict[str, Callable[[Series], Any] | str] | None

    def _process_polygons(
        self,
        polygons: GeoDataFrame,
        mask_geom: Polygon,
    ) -> GeoDataFrame:
        def _make_centroid(geom: Polygon) -> Point:
            centroid = geom.centroid

            if geom.has_z:
                # TODO: calculate z as mean of vertices' z?
                centroid = Point(centroid.x, centroid.y, 0.0)

            return centroid

        gdf = polygons.loc[
            (polygons.geometry.area < self.polygon_min_area)
            & (polygons.geometry.centroid.within(mask_geom))
        ].copy()

        gdf.geometry = gdf.geometry.apply(_make_centroid)

        return gdf

    def execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: GeoDataFrame containing Points and/or Polygons.
            reference_data: May contain a Polygon GeoDataFrame with the key
                "mask". If so new centroids will not be created outside of its
                features.

        Returns:
        -------
            GeoDataFrame with generalized data, which contains the newly
            created centroids and all input features which were not turned into
            centroids. These can be distinguished by the value in the column of
            the name set by the feature_type_column attribute.

        Raises:
        ------
            GeometryTypeError: If data contains something other than point or
            polygon geometries or mask data contains non-polygon geometries.

        """
        if not check_gdf_geometry_type(data, ["Point", "Polygon"]):
            msg = (
                "GeneralizePointClustersAndPolygonsToCentroids "
                + "works only with Point and Polygon geometries"
            )
            raise GeometryTypeError(msg)

        if reference_data.get("mask") is not None and not check_gdf_geometry_type(
            reference_data["mask"],
            [
                "Polygon",
                "MultiPolygon",
            ],
        ):
            msg = "mask dataframe must only contain (Multi)Polygons"
            raise GeometryTypeError(msg)

        points, polygons = (
            data.loc[data.geometry.type == "Point"],
            data.loc[data.geometry.type == "Polygon"],
        )

        clusters_from_points = GeoDataFrame()
        if not points.empty:
            clusters_from_points = get_cluster_centroids(
                points,
                self.cluster_distance,
                self.unique_id_column,
                aggregation_functions=self.aggregation_functions,
            )

            clusters_from_points[self.identity_hash_column] = (
                clusters_from_points.old_ids.apply(
                    lambda ids: sha256("".join(sorted(ids)).encode()).hexdigest(),
                )
            )

            clusters_from_points[self.feature_type_column] = "centroid_from_point"

        if reference_data.get("mask") is not None:
            mask_gdf = reference_data["mask"]
            mask_geom = mask_gdf.geometry.union_all()
        else:
            # Create a mask which will include all geometries. This is slightly
            # less efficient, but makes this function more readable and less
            # complex.
            x_min, y_min, x_max, y_max = data.total_bounds
            mask_geom = box(x_min, y_min, x_max, y_max)

        clusters_from_polygons = GeoDataFrame()
        if not polygons.empty:
            clusters_from_polygons = self._process_polygons(polygons, mask_geom)
            clusters_from_polygons[self.feature_type_column] = "centroid_from_polygon"

        if not clusters_from_points.empty:
            clusters_from_points = clusters_from_points[
                clusters_from_points.geometry.within(mask_geom)
            ]
            ids_to_remove = list(chain.from_iterable(clusters_from_points["old_ids"]))

            points = points.loc[~points[self.unique_id_column].isin(ids_to_remove)]
            clusters_from_points = clusters_from_points.drop(
                columns=["old_ids"],
                axis=1,
            )

        if not clusters_from_polygons.empty:
            polygons = polygons.loc[
                (polygons.geometry.area >= self.polygon_min_area)
                | (~polygons.geometry.centroid.within(mask_geom)),
            ]

        points[self.feature_type_column] = "unchanged_point"
        polygons[self.feature_type_column] = "unchanged_polygon"

        return GeoDataFrame(
            concat([clusters_from_points, clusters_from_polygons, points, polygons]),
        )
