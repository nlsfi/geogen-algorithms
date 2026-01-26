#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from conftest import IntegrationTest

from geogenalg.application.generalize_clusters_to_centroids import (
    GeneralizePointClustersAndPolygonsToCentroids,
)
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "kmtk_id"


@pytest.mark.parametrize(
    (
        "input_layers",
        "control_layer",
        "algorithm",
    ),
    [
        (
            ["boulder_in_water"],
            "control_only_points",
            GeneralizePointClustersAndPolygonsToCentroids(
                cluster_distance=30.0,
                polygon_min_area=1000.0,
            ),
        ),
        (
            ["boulders_in_water"],
            "control_only_polygons",
            GeneralizePointClustersAndPolygonsToCentroids(
                cluster_distance=30.0,
                polygon_min_area=1000.0,
            ),
        ),
        (
            ["boulder_in_water", "boulders_in_water"],
            "control_mixed",
            GeneralizePointClustersAndPolygonsToCentroids(
                cluster_distance=30.0,
                polygon_min_area=1000.0,
            ),
        ),
        (
            ["boulder_in_water"],
            "control_only_points_aggfunc",
            GeneralizePointClustersAndPolygonsToCentroids(
                cluster_distance=30.0,
                polygon_min_area=1000.0,
                aggregation_functions={
                    "boulder_in_water_type_id": lambda values: min(values)
                    if len(values) == set(values)
                    else 2
                },
            ),
        ),
    ],
    ids=[
        "only_points",
        "only_polygons",
        "mixed_geom_types",
        "only_points_aggfunc",
    ],
)
def test_generalize_boulders_in_water(
    testdata_path: Path,
    input_layers: list[str],
    control_layer: str,
    algorithm: GeneralizePointClustersAndPolygonsToCentroids,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "clusters_to_centroids.gpkg")

    inputs = [gpkg.to_input(layer) for layer in input_layers]

    IntegrationTest(
        input_uri=inputs,
        control_uri=gpkg.to_input(control_layer),
        algorithm=algorithm,
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
    ).run()
