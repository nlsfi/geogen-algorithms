#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from hashlib import sha256

from geopandas import GeoDataFrame
from pandas import Series, concat
from shapely import Point, Polygon
from sklearn.cluster import DBSCAN

from geogenalg.application import BaseAlgorithm
from geogenalg.cluster import get_cluster_centroids
from geogenalg.core.exceptions import GeometryTypeError


@dataclass(frozen=True)
class GeneralizeRocksToClusters(BaseAlgorithm):
    """Reduces polygons and point clusters to single points."""

    """Points within this distance of each other will be clustered."""
    cluster_distance: float
    """Polygons with area smaller than this will be turned into a point."""
    polygon_min_area: float
    """Name of column containing a unique identifier of input features."""
    unique_id_column: str

    def _process_polygons(self, polygons: GeoDataFrame) -> GeoDataFrame:
        def _make_centroid(geom: Polygon) -> Point:
            centroid = geom.centroid

            if geom.has_z:
                # TODO: calculate z as mean of vertices' z?
                centroid = Point(centroid.x, centroid.y, 0.0)

            return centroid

        gdf = polygons[polygons.geometry.area < self.polygon_min_area]
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
            reference_data: If contains a item "mask", new clusters will not be
                generated outside of it.

        Returns:
        -------
            GeoDataFrame with generalized data. FIXME: explain more

        Raises:
        ------
            GeometryTypeError: If data contains something other than point or
            polygon geometries

        """
        if "LineString" in data.geometry.type.to_numpy():
            # TODO: what about multigeometries/geom collections?
            msg = "only points or polygons allowed"
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
            )

            clusters_from_points[self.identity_hash_column] = (
                clusters_from_points.old_ids.apply(
                    lambda ids: sha256("".join(ids).encode()).hexdigest(),
                )
            )

            clusters_from_points = clusters_from_points.drop(
                columns=["old_ids"],
                axis=1,
            )

        clusters_from_polygons = GeoDataFrame()
        if not polygons.empty:
            clusters_from_polygons = self._process_polygons(polygons)

        return GeoDataFrame(concat([clusters_from_points, clusters_from_polygons]))


def generalize_boulders_in_water(
    point_gdf: GeoDataFrame,
    mask_gdf: GeoDataFrame,
    cluster_distance: float,
    polygon_gdf: GeoDataFrame | None = None,
    polygon_min_area: float | None = None,
) -> dict[str, GeoDataFrame]:
    """Generalize boulders in water.

    Generalizes boulders in water in a point dataset nearby points
    within the given cluster_distance will be turned into a new feature in
    a separate point feature class "boulder cluster" at the cluster's centroid.
    Polygonal water boulder areas will be turned into a point boulder cluster
    at the polygon's centroid if their size is under the given polygon_min_area.

    The newly generated boulder clusters must be within the polygonal mask
    dataset, otherwise the features will not be turned into clusters.

    Returns:
        Dictionary containing clustering results.

    """

    def aggregate_boulder_in_water_type(ls: Series) -> int:
        if len(set(ls)) == 1:
            return int(ls.to_numpy()[0])

        return 2

    def aggregate_fid(ls: Series) -> list[int]:
        return ls.tolist()

    cid = "cid"
    boulder_in_water_type = "boulder_in_water_type_id"
    type_column = "boulder_cluster_in_water_type_id"

    # cluster points
    coords = point_gdf.get_coordinates().to_numpy()
    cluster_labels = DBSCAN(eps=cluster_distance, min_samples=2).fit(coords).labels_  # noqa: SC200
    labels = Series(cluster_labels).rename(cid)

    gdf = GeoDataFrame(concat([point_gdf, labels], axis=1))
    gdf = gdf.reset_index()
    gdf = gdf.rename(columns={"index": "fid"})

    clusters: GeoDataFrame = gdf.dissolve(
        cid,
        aggfunc={  # noqa: SC200
            boulder_in_water_type: aggregate_boulder_in_water_type,
            "fid": aggregate_fid,
        },
        as_index=False,
    )

    clusters = clusters[clusters[cid] != -1]
    clusters.geometry = clusters.geometry.centroid
    clusters = clusters.rename(columns={boulder_in_water_type: type_column})
    clusters = clusters[clusters.geometry.within(mask_gdf.unary_union)]

    clustered_fids = [fid for fids in clusters["fid"].tolist() for fid in fids]
    new_point_gdf = point_gdf.reset_index().rename(columns={"index": "fid"})
    new_point_gdf = new_point_gdf[~new_point_gdf["fid"].isin(clustered_fids)]

    clusters = clusters.drop("fid", axis=1)

    results = {"boulder_points": new_point_gdf}

    if polygon_gdf is not None:
        poly_clusters = polygon_gdf[polygon_gdf.geometry.area < polygon_min_area]
        poly_clusters.geometry = poly_clusters.geometry.centroid
        poly_clusters = poly_clusters[
            poly_clusters.geometry.within(mask_gdf.unary_union)
        ]
        poly_clusters = poly_clusters[["geometry"]]
        poly_clusters[type_column] = 2

        clusters = concat([clusters, poly_clusters])

        results["boulder_polygons"] = polygon_gdf[
            ~polygon_gdf.index.isin(poly_clusters.index)
        ]

    clusters = clusters.drop(cid, axis=1)

    results["clusters"] = clusters

    return results
