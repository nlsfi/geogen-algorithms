# ruff: noqa: F841
from qgis.core import QgsGeometry, QgsPointXY, QgsVectorLayer


def calculate_sinuosity(geom: QgsGeometry) -> float:
    # if (geom.wkbType() != 2 and geom.wkbType != 1002):
    #    raise Exception("Input must be a line geometry, not: ", geom.wkbType())

    line_length = geom.length()
    line_nodes = geom.asPolyline()

    first_point = QgsPointXY(line_nodes[0])
    last_point = QgsPointXY(line_nodes[-1])
    straight_line = QgsGeometry.fromPolylineXY(
        [
            first_point,
            last_point,
        ]
    )
    straight_distance = straight_line.length()
    return line_length / straight_distance


def check_line_connectivity(
    geom: QgsGeometry, layer: QgsVectorLayer, radius: float
) -> bool:
    line_nodes = geom.asPolyline()
    first_node = QgsPointXY(line_nodes[0])
    end_node = QgsPointXY(line_nodes[-1])
    first_point_geom = QgsGeometry.fromPointXY(first_node)
    end_point_geom = QgsGeometry.fromPointXY(end_node)

    first_node_buffer = first_point_geom.buffer(radius, 5)
    end_node_buffer = end_point_geom.buffer(radius, 5)
    first_node_buffer_geometry_engine = QgsGeometry.createGeometryEngine(
        first_node_buffer.constGet()
    )
    end_node_buffer_geometry_engine = QgsGeometry.createGeometryEngine(
        end_node_buffer.constGet()
    )

    first_node_buffer_geometry_engine.prepareGeometry()
    end_node_buffer_geometry_engine.prepareGeometry()
    intersecting_items_count = 0
    for feature in layer.getFeatures():
        feature_geom = feature.geometry()
        if first_node_buffer_geometry_engine.intersects(feature_geom.constGet()):
            intersecting_items_count += 1
        if end_node_buffer_geometry_engine.intersects(feature_geom.constGet()):
            intersecting_items_count += 1

    # Both node buffers intersect with feature itself which gives 2
    # intersections
    return not intersecting_items_count < 4  # noqa: PLR2004
