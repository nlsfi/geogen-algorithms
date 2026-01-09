#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import Any

from geopandas.geodataframe import GeoDataFrame
from pandas import Series, concat
from shapely import MultiPoint, Point
from sklearn.cluster import DBSCAN

from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.core.geometry import mean_z


def dbscan_cluster_ids(
    input_gdf: GeoDataFrame,
    cluster_distance: float,
    *,
    cluster_column_name: str = "cluster_id",
    ignore_z: bool = True,
) -> Series:
    """Create Series of cluster ids.

    Args:
    ----
        input_gdf: GeoDataFrame containing Points to be clustered.
        cluster_distance: Points within this distance will be considered a
            cluster, must be > 0.0.
        cluster_column_name: Name of the returned Series.
        ignore_z: Whether the point's z coordinate should affect clustering.

    Returns:
    -------
        Series of int64 denoting which cluster the corresponding point belongs
        to (-1 meaning the point does not belong to any cluster).

    """
    clusters = DBSCAN(
        eps=cluster_distance,  # noqa: SC200
        min_samples=2,
    ).fit(input_gdf.get_coordinates(include_z=not ignore_z).to_numpy())

    return Series(clusters.labels_, name=cluster_column_name)


def get_cluster_centroids(
    input_gdf: GeoDataFrame,
    cluster_distance: float,
    *,
    aggregation_functions: dict[str, Callable[[Series], Any] | str] | None = None,
    old_ids_column: str = "old_ids",
) -> GeoDataFrame:
    """Cluster points in a GeoDataFrame and return their centroids.

    Args:
    ----
        input_gdf: GeoDataFrame containing Points to be clustered.
        cluster_distance: Points within this distance will be clustered, must
            be > 0.0.
        aggregation_functions: Dictionary of aggregation functions, which is
            passed to geopandas' dissolve function. The keys should correspond
            to column names in the input GeoDataFrame. If aggregation function
            is not specified for column, "first" will be used.
        old_ids_column: Name of the column in the output GeoDataFrame
            containing a tuple of the cluster's old identifiers.

    Returns:
    -------
        A new GeoDataFrame containing the centroids of each cluster and
        identifiers of the original points as a tuple in a new column. If the
        input points have a z value, the centroid's z value will be the mean of
        all the points in the cluster.

    """
    gdf = input_gdf.copy()

    cluster_id_column = "__cluster_id"  # temporary column
    cluster_ids = (
        dbscan_cluster_ids(
            gdf,
            cluster_distance,
            cluster_column_name=cluster_id_column,
        )
        .to_frame()
        .set_index(gdf.index)
    )

    gdf[cluster_id_column] = cluster_ids[cluster_id_column]

    gdf = gdf.loc[gdf[cluster_id_column] != -1]

    # Copy id column, so all the cluster's ids can be aggregated into it later
    gdf[old_ids_column] = gdf.index

    if aggregation_functions is None:
        aggregation_functions = {}

    # If the aggfunc parameter is passed into dissolve() and columns are
    # missing, they are dropped. Therefore explicitly define a default
    # aggregation function for each column if one was not already given.
    for column in gdf:
        if column in {gdf.geometry.name, old_ids_column, cluster_id_column}:
            continue

        if column not in aggregation_functions:
            aggregation_functions[str(column)] = "first"

    aggregation_functions[old_ids_column] = lambda ids: tuple(ids.to_list())

    gdf = gdf.dissolve(
        by=cluster_id_column,
        as_index=False,
        aggfunc=aggregation_functions,  # noqa: SC200
    )
    gdf = gdf.drop(columns=[cluster_id_column])

    # Dissolve aggregates geometries into a MultiPoint, turn into single point
    # at its centroid.
    def _make_centroid(geom: MultiPoint) -> Point:
        centroid = geom.centroid

        if geom.has_z:
            centroid = Point(centroid.x, centroid.y, mean_z(geom))

        return centroid

    gdf.geometry = gdf.geometry.apply(_make_centroid)

    return gdf


def reduce_nearby_points_by_clustering(
    input_gdf: GeoDataFrame,
    reduce_threshold: float,
    unique_key_column: str,
    cluster_members_column: str = "cluster_members",
) -> GeoDataFrame:
    """Reduce the number of points by clustering and replacing them with their centroid.

    Note:
        The Z coordinate of each new point is calculated as the mean of the Z values of
        the clustered points. If none of the clustered points have a Z coordinate, the Z
        value is set to 0.0.

    Args:
    ----
        input_gdf: Input GeoDataFrame with point geometries. The GeoDataFrame must
              include a column with a unique key.
        reduce_threshold: Distance used for buffering and clustering
        unique_key_column: Name of the column containing unique identifiers
        cluster_members_column: Name of the column that lists the points included in
              each cluster.

    Returns:
    -------
        A GeoDataFrame containing centroids of clusters.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than point features

    """
    if input_gdf.geometry.type.unique().tolist() != ["Point"]:
        msg = "reduce_nearby_points only supports Point geometries."
        raise GeometryTypeError(msg)

    buffered_gdf = input_gdf.copy()

    buffered_gdf.geometry = buffered_gdf.buffer(reduce_threshold)

    dissolved_gdf: GeoDataFrame = (
        buffered_gdf.dissolve().explode(index_parts=True).reset_index(drop=True)
    )

    clustered_points_gdfs: list[GeoDataFrame] = []

    for cluster_polygon in dissolved_gdf.geometry:
        in_cluster_gdf = input_gdf[input_gdf.geometry.within(cluster_polygon)]

        min_id = in_cluster_gdf[unique_key_column].min()
        representative_point_gdf: GeoDataFrame = in_cluster_gdf.loc[
            in_cluster_gdf[unique_key_column] == min_id
        ].copy()

        # Shapely/GeoPandas centroid discards Z-dimension
        xy_centroid = cluster_polygon.centroid

        # Estimate Z coordinates for centroids by averaging Z values of clustered points
        z_values = [
            geom.z
            for geom in in_cluster_gdf.geometry
            if isinstance(geom, Point) and geom.has_z
        ]
        average_z = sum(z_values) / len(z_values) if z_values else 0.0

        new_centroid = Point(xy_centroid.x, xy_centroid.y, average_z)
        representative_point_gdf = representative_point_gdf.set_geometry([new_centroid])

        # If several points are clustered, their keys are saved as cluster members
        if len(in_cluster_gdf) > 1:
            new_cluster_members = in_cluster_gdf[unique_key_column].tolist()
            if (
                cluster_members_column in representative_point_gdf.columns
                and representative_point_gdf[cluster_members_column] is not None
            ):
                existing_members = representative_point_gdf[cluster_members_column]
                representative_point_gdf[cluster_members_column] = (
                    existing_members + new_cluster_members
                )

            else:
                representative_point_gdf[cluster_members_column] = [new_cluster_members]

        # For single points, the cluster members field is set to None if the field
        # does not contain any cluster members
        elif cluster_members_column not in representative_point_gdf.columns:
            representative_point_gdf[cluster_members_column] = None

        clustered_points_gdfs.append(representative_point_gdf)

    return concat(clustered_points_gdfs)
