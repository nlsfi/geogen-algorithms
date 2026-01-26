#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
from geopandas import GeoDataFrame
from pandas.testing import assert_series_equal
from shapely.geometry import MultiPolygon, Point, Polygon

from geogenalg.application.remove_overlap import RemoveOverlap
from geogenalg.core.exceptions import (
    GeometryTypeError,
    MissingReferenceError,
)


def test_remove_overlap():
    input_polygons = [
        Polygon([(0, 0, 1), (0, 4, 1), (4, 4, 1), (4, 0, 1)]),
        Polygon([(10, 0, 5), (10, 2, 5), (8, 2, 5), (8, 0, 5)]),  # untouched polygon
    ]
    input_gdf = GeoDataFrame(
        {"id": [1122, 2233], "name": ["A", "B"], "value": [10, 20]},
        geometry=input_polygons,
        crs="EPSG:4326",
    )
    input_gdf.set_index("id")

    algorithm = RemoveOverlap(reference_key="mask")

    # CASE 1: Mask polygon overlaps a section of input polygon
    mask_poly = Polygon([(0, 3), (0, 4), (4, 4), (4, 3)])
    mask_gdf = GeoDataFrame(
        {"id": [1], "foo": ["bar"]}, geometry=[mask_poly], crs="EPSG:4326"
    )

    out = algorithm.execute(input_gdf, {"mask": mask_gdf})
    assert len(out) == 2
    assert type(out[out["name"] == "A"].iloc[0].geometry) is Polygon
    assert type(out[out["name"] == "B"].iloc[0].geometry) is Polygon

    # Check that polygon B from input is untouched since it did not overlap with mask polygon
    out_b = out[out["name"] == "B"].iloc[0].geometry
    original_b = input_gdf[input_gdf["name"] == "B"].iloc[0].geometry
    assert out_b == original_b

    # Check that polygon A that was cut has smaller area now
    out_a = out[out["name"] == "A"].iloc[0].geometry
    original_a = input_gdf[input_gdf["name"] == "A"].iloc[0].geometry
    assert out_a.area < original_a.area

    # Verify that Z value (1) is preserved for unchanged vertices
    assert {round(z, 3) for _, _, z in out_a.exterior.coords} == {1}

    # CASE 2: Mask polygon overlaps a section of input polygon so that it gets split
    mask_poly_2 = Polygon([(0, 2), (0, 3), (4, 3), (4, 2)])
    mask_gdf_2 = GeoDataFrame(
        {"id": [1], "foo": ["bar"]}, geometry=[mask_poly_2], crs="EPSG:4326"
    )

    out = algorithm.execute(input_gdf, {"mask": mask_gdf_2})
    assert (
        type(out[out["name"] == "A"].iloc[0].geometry) is MultiPolygon
    )  # Polygon was split into parts

    # Value column should be preserved from input
    assert_series_equal(out["value"], input_gdf["value"])


def test_remove_overlap_invalid_input_geometry():
    with pytest.raises(
        GeometryTypeError,
        match=r"Input data must contain only geometries of following types: Polygon.",
    ):
        RemoveOverlap().execute(
            data=GeoDataFrame(
                {
                    "id": [1],
                },
                geometry=[Point(0, 0)],
                crs="EPSG:4326",
            ),
            reference_data={"mask": GeoDataFrame()},
        )


def test_remove_overlap_missing_reference_data():
    with pytest.raises(
        MissingReferenceError,
        match=r"Reference data is missing.",
    ):
        RemoveOverlap().execute(
            data=GeoDataFrame(
                {
                    "id": [1],
                },
                geometry=[Polygon([(0, 0), (0, 1), (1, 1)])],
                crs="EPSG:4326",
            ),
            reference_data={},
        )


def test_remove_overlap_invalid_mask_geometry():
    with pytest.raises(
        GeometryTypeError,
        match=r"Reference data must contain only geometries of following types: Polygon.",
    ):
        RemoveOverlap().execute(
            data=GeoDataFrame(
                {
                    "id": [1],
                },
                geometry=[Polygon([(0, 0), (0, 1), (1, 1)])],
                crs="EPSG:4326",
            ),
            reference_data={
                "mask": GeoDataFrame(
                    {
                        "id": [1],
                    },
                    geometry=[Point(0, 0)],
                    crs="EPSG:4326",
                )
            },
        )
