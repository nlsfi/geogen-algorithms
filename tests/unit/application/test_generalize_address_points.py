#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from pathlib import Path

from conftest import IntegrationTest

from geogenalg.application.generalize_address_points import GeneralizeAddressPoints
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "id"


def test_generalize_address_points(
    testdata_path: Path,
) -> None:
    gpkg = GeoPackagePath(testdata_path / "addresses.gpkg")

    algorithm = GeneralizeAddressPoints(
        join_column="permanent_building_identifier",
        reference_join_column="vtj_prt",
        priority_column="address_number",
        duplicate_location_tolerance=1.0,
        max_building_distance=200.0,
        search_radius=500.0,
        reference_key="buildings",
    )

    IntegrationTest(
        input_uri=gpkg.to_input("address_point"),
        control_uri=gpkg.to_input("address_point_control"),
        algorithm=algorithm,
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
        reference_uris={
            "buildings": gpkg.to_input("building_part_area"),
        },
        assert_function_arguments={
            "check_less_precise": True,
        },
        dummy_data_mandatory_columns=frozenset(
            [
                algorithm.join_column,
                algorithm.priority_column,
                "number_part_of_address_number",
                "address_fin",
                "address_swe",
            ]
        ),
        dummy_reference_data_mandatory_columns=frozenset(
            [
                algorithm.reference_join_column,
                "building_function_id",
            ]
        ),
    ).run()
