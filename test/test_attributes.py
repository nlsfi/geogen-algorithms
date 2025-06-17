#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from geogenalg import attributes


@pytest.mark.parametrize(
    ("source_gdf", "target_gdf", "expected_attribute_value"),
    [
        (
            gpd.GeoDataFrame(
                {"id": [1], "mtk_id": ["3491230"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                crs="EPSG:3067",
            ),
            "3491230",
        ),
        (
            gpd.GeoDataFrame(
                {"id": [1, 2, 3], "mtk_id": ["3491230", "344531", "1234"]},
                geometry=[
                    Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 1)]),
                    Polygon([(1, 1), (1.5, 1), (1.5, 1.5), (1, 1.5)]),
                    Polygon([(0.9, 0.9), (1.9, 0.9), (1.9, 1.9), (0.9, 1.9)]),
                ],
                crs="EPSG:3067",
            ),
            gpd.GeoDataFrame(
                {"id": [4], "mtk_id": ["0000"]},
                geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                crs="EPSG:3067",
            ),
            ["3491230", "1234"],
        ),
    ],
    ids=["same_polygon", "two_partly_intersecting_polygons"],
)
def test_inherit_attributes(
    source_gdf: gpd.GeoDataFrame,
    target_gdf: gpd.GeoDataFrame,
    expected_attribute_value: str,
):
    result = attributes.inherit_attributes(source_gdf, target_gdf)
    assert "mtk_id" in result.columns
    assert result.iloc[0]["mtk_id"] in expected_attribute_value
    assert len(result) == len(target_gdf)
