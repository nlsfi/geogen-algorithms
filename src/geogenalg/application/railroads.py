from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from math import sqrt
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN


def generalize_railroads(
    railtracks_gdf: gpd.GeoDataFrame,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Generalizes railroad network by identifying different railroad track types
    (sidetracks & maintracks, for example) and reducing and them in dense
    areas. Influenced largely by the proposed algorithm by Savino & Touya
    (hereafter S&T) in (pdf, open access):
    www.researchgate.net/publication/281857051_Automatic_Structure_Detection_and_Generalization_of_Railway_Networks

    Parameters:
        railtracks_gdf (GeoDataFrame): input railtracks GeoDataFrame with
        LineString geometry.
    """

    # The basic steps of this algorithm are
    # 1a) Create an orthogonal line to the middle point of every input line and
    # determine the intersections of this line with the input lines. If it has
    # many intersections it is a candidate to belong to a sidetrack group.
    # Otherwise, it is a maintrack (S&T 2.2).
    # 1b) Create orthogonal lines to each side of the input feature and count
    # the intersections in order to detect if a track is on the border (this
    # step is not in S&T). Border tracks will not be generalized away
    # 2) Cluster sidetrack candidates for proximity (DBSCAN). Cluster with
    # many tracks forms a sidetrack group
    # 3) Record connections between sidetrack clusters and maintracks
    # 4) Form strokes within the main tracks (NOT YET IMPLEMENTED! S&T 2.3.1)
    # 5) Form maintrack groups iteratively
    # 6) Record the line ending types of maintracks in each group
    # 7) Generalize maintracks by choosing only the longest in each group
    # 8) Fix the line endings to preserve topology (e.g. if generalization
    # altered a line ending type, NOT YET IMPLEMENTED!)
    # 9) Detect structures inside sidetrack groups: fans, packs, free tracks
    # (NOT IMPLEMENTED! S&T 2.3.2))
    # 10) Generalize sidetracks by removing with fixed ratio in each group (Would
    # be handled differently in S&T once step 9 is implemented)

    generalized_gdf = generalize_railtracks(railtracks_gdf)

    return {"railtracks": generalized_gdf}


def calculate_intersections_between_columns(
        gdf: gpd.GeoDataFrame,
        geom_col1="geometry",
        geom_col2="other_geometry",
        intersection_count_field="intersection_count"
        ) -> gpd.GeoDataFrame:
    """
    Calculate the number of intersections between two geometry columns in a
    GeoDataFrame and record the result.
    """
    # Ensure both geometry columns exist
    if geom_col1 not in gdf.columns or geom_col2 not in gdf.columns:
        msg = f"Both '{geom_col1}' and '{geom_col2}' must be in the GeoDataFrame."
        raise ValueError(msg)

    # Ensure geometries are valid
    gdf = gdf[gdf[geom_col1].is_valid & gdf[geom_col2].is_valid]

    # Create spatial index for geometry_col1
    spatial_index = gdf.sindex

    # Calculate intersections for each row
    intersection_counts = []
    for idx, row in gdf.iterrows():
        geom1 = row[geom_col1]
        geom2 = row[geom_col2]

        # Find potential matches for geom2 using the spatial index of geom1
        possible_matches_index = list(spatial_index.intersection(geom2.bounds))

        # Check actual intersections
        actual_intersections = gdf.iloc[possible_matches_index][geom_col1].intersects(geom2)

        # Count the intersections
        intersection_count = actual_intersections.sum()
        intersection_counts.append(intersection_count)

    gdf[intersection_count_field] = intersection_counts

    return gdf


def create_orthogonal_lines(gdf: gpd.GeoDataFrame, length: float):
    def calculate_orthogonal_line(line, length):
        centroid = line.interpolate(0.5, normalized=True)

        # Get the line end points coordinates to calculate the bearing
        coords = list(line.coords)
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]

        # Calculate the angle of the orthogonal line (i.e. rotated 90 degrees)
        angle = np.arctan2(dy, dx) + np.pi / 2

        # Calculate half of the length for the orthogonal line
        dx_ortho = (length / 2) * np.cos(angle)
        dy_ortho = (length / 2) * np.sin(angle)

        # Create the orthogonal line
        p1 = Point(centroid.x - dx_ortho, centroid.y - dy_ortho)
        p2 = Point(centroid.x + dx_ortho, centroid.y + dy_ortho)
        return LineString([p1, p2])

    # Apply the function to each line in the GeoDataFrame
    gdf["orthogonal_line"] = gdf.geometry.apply(lambda line: calculate_orthogonal_line(line, length))
    return gdf


def create_orthogonal_left_sidelines(gdf: gpd.GeoDataFrame, length: float) -> gpd.GeoDataFrame:
    def calculate_orthogonal_line_left(line, length):
        centroid = line.interpolate(0.5, normalized=True)

        # Get the coordinates of the first two points to calculate the bearing
        coords = list(line.coords)
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]

        # Calculate the angle of the orthogonal line (90 degrees rotated)
        angle = np.arctan2(dy, dx) + np.pi / 2

        # Create the orthogonal line
        p1 = Point(centroid.x, centroid.y)
        p2 = Point(centroid.x + length*np.cos(angle), centroid.y + length*np.sin(angle) )
        return LineString([p1, p2])

    # Apply the function to each line in the GeoDataFrame
    gdf["orthogonal_line_left"] = gdf.geometry.apply(lambda line: calculate_orthogonal_line_left(line, length))
    return gdf


def create_orthogonal_right_sidelines(gdf, length):
    def calculate_orthogonal_line_right(line, length):
        # Calculate the centroid of the line
        centroid = line.interpolate(0.5, normalized=True)

        # Get the coordinates of the first two points to calculate the bearing
        coords = list(line.coords)
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]

        # Calculate the angle of the orthogonal line (90 degrees rotated)
        angle = np.arctan2(dy, dx) - np.pi / 2

        # Create the orthogonal line
        p1 = Point(centroid.x, centroid.y)
        p2 = Point(centroid.x + length*np.cos(angle), centroid.y + length*np.sin(angle) )
        return LineString([p1, p2])

    # Apply the function to each line in the GeoDataFrame
    gdf["orthogonal_line_right"] = gdf.geometry.apply(lambda line: calculate_orthogonal_line_right(line, length))
    return gdf


def reduce_line_density(gdf: gpd.GeoDataFrame, distance_threshold=5, sampling_rate=0.5):
    """
    Remove parallel lines while trying to main the local fluctuations in the density.
    Distance_threshold: Max. distance for clustering parallel lines.
    Sampling_rate: Proportion of lines to keep in each cluster (0 < sampling_rate <= 1).
    """
    # Check that the geometry column contains only LineStrings
    if not all(gdf.geometry.type == "LineString"):
        msg = "GeoDataFrame should contain only LineStrings!"
        raise ValueError(msg)

    # Extract centroids of each line for clustering
    centroids = gdf.geometry.apply(lambda line: line.centroid)
    coords = np.array([[point.x, point.y] for point in centroids])

    # Cluster lines based on proximity using DBSCAN
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(coords)
    gdf['cluster'] = clustering.labels_

    # Reduce density within each cluster
    reduced_lines = []
    for cluster_id in gdf['cluster'].unique():
        cluster_lines = gdf[gdf['cluster'] == cluster_id]

        # Randomly sample lines within the cluster
        # Set random_state for reproducibility
        sampled_lines = cluster_lines.sample(
            frac=sampling_rate, random_state=42
        )
        reduced_lines.append(sampled_lines)

    # Combine all sampled lines into a single GeoDataFrame
    reduced_gdf = gpd.GeoDataFrame(pd.concat(reduced_lines, ignore_index=True),
                                   crs=gdf.crs).drop(columns=['cluster'])

    return reduced_gdf  # noqa


def generalize_sidetracks(sidetracks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Simplified sidetrack generalization approach:
    # select representative tracks
    inner_sidetracks = sidetracks[~sidetracks['border']]
    outer_sidetracks = sidetracks[sidetracks['border']]
    removed_dense_tracks = inner_sidetracks[::5]  # select every 5th
    gen_sidetracks = gpd.GeoDataFrame(
        pd.concat([removed_dense_tracks, outer_sidetracks]), crs=sidetracks.crs
        )
    return gen_sidetracks  # noqa


def collapse_parallel_tracks(parallel_groups: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Filter out invalid geometries and ensure only LineStrings are processed
    valid_lines = [
        line for line in parallel_groups.geometry if line and isinstance(line, LineString)
        ]
    if not valid_lines:
        return gpd.GeoDataFrame({'geometry': []})
    # Calculate an approximate center line for the parallel tracks
    all_coords = np.vstack([list(line.coords) for line in valid_lines])
    mean_coords = np.mean(all_coords, axis=0)
    # Ensure that the mean_coords can form a valid LineString
    if len(mean_coords) < 4:  # Need at least two (x,y) points
        return gpd.GeoDataFrame({'geometry': []})
    center_line = LineString([Point(coord) for coord in mean_coords.reshape(-1, 2)])
    return gpd.GeoDataFrame({'geometry': [center_line]})


def generalize_railtracks(
        gdf: gpd.GeoDataFrame,
        max_intersections=7,
        border_intersections=2,
        distance_threshold=50,
        min_samples=3,
        ) -> gpd.GeoDataFrame:
    """
    max_intersections (int): how many intersections of the orthogonal line drawn in
        the middle point of the track can a rail have before it is considered
        belonging to a sidetrack group
    border_intersections (int): How many intersections can the one-sided orthogonal
        line (otherwise similar to the orthogonal line) have before it is
        considered to be a track on the border (of maybe a larger sidetrack
        group)
    distance_threshold (float):
    min_samples (int):
    """

    # STEP 1

    gdf = create_orthogonal_lines(gdf, 100)
    gdf = create_orthogonal_left_sidelines(gdf, 30)
    gdf = create_orthogonal_right_sidelines(gdf, 30)

    gdf = calculate_intersections_between_columns(gdf, geom_col1="geometry", geom_col2="orthogonal_line", intersection_count_field="intersection_count")
    gdf = calculate_intersections_between_columns(gdf, geom_col1="geometry", geom_col2="orthogonal_line_left", intersection_count_field="intersections_left")
    gdf = calculate_intersections_between_columns(gdf, geom_col1="geometry", geom_col2="orthogonal_line_right", intersection_count_field="intersections_right")

    gdf['border_all'] = (gdf['intersections_left'] < border_intersections) | ( gdf['intersections_right'] < border_intersections)

    gdf['sidetrack_candidate'] = gdf["intersection_count"] >= max_intersections

    gdf_sidetracks = gdf[gdf['sidetrack_candidate']]
    gdf_maintracks = gdf[~gdf['sidetrack_candidate']]

    # STEP 2

    centroids = gdf_sidetracks.geometry.apply(lambda line: line.centroid)
    coords = np.array([[point.x, point.y] for point in centroids])

    clustering = DBSCAN(eps=distance_threshold, min_sample=min_samples).fit(coords)
    gdf_sidetracks['cluster'] = clustering.labels_

    combined_tracks = gpd.GeoDataFrame(
        pd.concat([gdf_maintracks, gdf_sidetracks]),
        crs=gdf.crs,
        )

    # STEP 3

    maintracks = combined_tracks[~combined_tracks['sidetrack_candidate']].copy()
    sidetracks = combined_tracks[combined_tracks['sidetrack_candidate']].copy()

    maintrack_touching_clusters = {}

    # Iterate over maintracks
    for idx, maintrack in maintracks.iterrows():
        touching_sidetracks = sidetracks[sidetracks.geometry.touches(maintrack.geometry)]

        touching_clusters = touching_sidetracks['cluster'].unique().tolist()
        maintrack_touching_clusters[idx] = touching_clusters

    # Add the new field to the maintracks GeoDataFrame
    maintracks['touches_clusters'] = maintracks.index.map(maintrack_touching_clusters)

    # Merge back into the original GeoDataFrame
    combined_tracks = gpd.GeoDataFrame(
        pd.concat([combined_tracks.drop(maintracks.index), maintracks]),
        crs=gdf.crs
        )

    """
    TODO: STEP 4: Implement strokes algorithm for railtracks and apply it to
    maintracks: output maintrack_strokes_gdf

    # STEP 5:

    gdf = maintrack_strokes_gdf
    gdf['stroke_length'] = gdf['geometry'].length
    gdf = gdf.sort_values(by='stroke_length', ascending=False).reset_index(drop=True)
    # Buffer distance around the central line
    buffer_distance = 10
    # Minimum intersection length for the track to be included in group
    min_intersection_length = 100

    # Column to store group IDs, initialized with -1 (ungrouped)
    gdf['group_id'] = -1
    current_group_id = 0

    # Iterate until all lines are grouped
    # the latter condition is to ensure that we do not end up in an infinite
    # loop. Might be better to replace it with some condition that terminates
    # if there is no change in group_ids
    while (gdf['group_id'] == -1).any() and current_group_id < 5000:
        # Select the longest remaining ungrouped line as the central line
        central_line = gdf[gdf['group_id'] == -1].iloc[0]
        buffer = central_line['geometry'].buffer(buffer_distance)

        # Find lines that intersect with the buffer
        potential_group = gdf[(gdf['group_id'] == -1) & (gdf['geometry'].intersects(buffer))]

        # Check if the intersection length is above the threshold
        def intersection_length(line, buffer):
            intersection = line.intersection(buffer)
            if intersection.is_empty:
                return 0
            if isinstance(intersection, LineString):
                return intersection.length
            return sum(geom.length for geom in intersection.geoms if isinstance(geom, LineString))

        # Assign group_id to lines with sufficient intersection length
        for idx, row in potential_group.iterrows():
            if intersection_length(row['geometry'], buffer) >= min_intersection_length:
                gdf.at[idx, 'group_id'] = current_group_id

        # Move to the next group
        current_group_id += 1
        """

    """
    # STEP 6:

    def classify_line_relation(
            line: LineString,
            central_line: LineString,
            converge_dist=1,
            diverge_dist=25
            ):
        line_start, line_end = Point(line.coords[0]), Point(line.coords[-1])
        # Get start and end points of the central line
        central_start, central_end = Point(central_line.coords[0]), Point(central_line.coords[-1])

        # Calculate distances between line ends and central line ends
        start_to_start = line_start.distance(central_start)
        start_to_end = line_start.distance(central_end)
        end_to_start = line_end.distance(central_start)
        end_to_end = line_end.distance(central_end)

        # Determine the start relation
        if min(start_to_start, start_to_end) <= converge_dist:
            start_relation = 'converges'
        elif min(start_to_start, start_to_end) >= diverge_dist:
            start_relation = 'diverges'
        else:
            start_relation = 'parallel'

        # Determine the end relation
        if min(end_to_start, end_to_end) <= converge_dist:
            end_relation = 'converges'
        elif min(end_to_start, end_to_end) >= diverge_dist:
            end_relation = 'diverges'
        else:
            end_relation = 'parallel'

        return start_relation, end_relation

    step6_input_gdf['start_relation'], step6_input_gdf['end_relation'] = zip(*step6_input_gdf.apply(
    lambda row: classify_line_relation(row['geometry'], central_lines[row['group']]), axis=1
    """

    """
    # STEP 7:

    def sort_groups_by_length(gdf):
        return gdf.sort_values(by='stroke_length', ascending=False).reset_index(drop=True

    def get_longest_line_per_group(gdf, group_col='group_id', length_col='stroke_length'):
        # Calculate line lengths if not already present
        if length_col not in gdf.columns:
            gdf[length_col] = gdf['geometry'].length

        # Sort by length within each group and select the longest line per group
        longest_lines = gdf.sort_values(length_col, ascending=False).drop_duplicates(group_col)

        return longest_lines

    maintracks_generalized_to_longest = get_longest_line_per_group(step7_input_gdf_maintracks)

    # STEP 10:

    # Simplified version to collapse_paralle_tracks:
    sidetracks_generalized = generalize_sidetracks(step7_input_gdf_sidetracks)

    """
