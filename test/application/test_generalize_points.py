#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from conftest import IntegrationTest

from geogenalg.application.generalize_points import GeneralizePoints
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "kmtk_id"


@pytest.mark.parametrize(
    (
        "input_layer",
        "control_layer",
        "algorithm",
    ),
    [
        (
            "boulder_in_water",
            "control_no_displacement",
            GeneralizePoints(
                cluster_distance=30.0,
                displace=False,
                displace_threshold=70.0,
                displace_points_iterations=10,
                aggregation_functions=None,
                is_cluster_column="is_cluster",
            ),
        ),
        (
            "boulder_in_water",
            "control_aggfunc",
            GeneralizePoints(
                cluster_distance=30.0,
                displace=False,
                displace_threshold=70.0,
                displace_points_iterations=10,
                aggregation_functions={
                    "boulder_in_water_type_id": lambda values: min(values)
                    if len(values) == set(values)
                    else 2
                },
                is_cluster_column="is_cluster",
            ),
        ),
        (
            "boulder_in_water",
            "control_displacement",
            GeneralizePoints(
                cluster_distance=30.0,
                displace=True,
                displace_threshold=70.0,
                displace_points_iterations=10,
                aggregation_functions=None,
                is_cluster_column="is_cluster",
            ),
        ),
    ],
    ids=[
        "points",
        "aggfunc",
        "displacement",
    ],
)
def test_generalize_points(
    testdata_path: Path,
    input_layer: str,
    control_layer: str,
    algorithm: GeneralizePoints,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "points.gpkg")

    IntegrationTest(
        input_uri=gpkg.to_input(input_layer),
        control_uri=gpkg.to_input(control_layer),
        algorithm=algorithm,
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
        dummy_data_mandatory_columns=["boulder_in_water_type_id"],
    ).run()
