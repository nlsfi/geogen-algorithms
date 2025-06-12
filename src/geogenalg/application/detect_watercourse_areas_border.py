# ruff: noqa: F841
#  ruff: noqa: SIM105

from __future__ import annotations

try:
    import processing  # noqa: F401
except ImportError:
    pass

from typing import Any

from qgis.core import (
    QgsFeature,
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QVariant,
)
from yleistys_qgis_plugin.core.utils import get_subalgorithm_result_layer


class DetectWatercourseAreasBorder(QgsProcessingAlgorithm):
    """The algorithm detects ...

    Parameters
    ----------
    INPUT: Vector layer
        The input layer that is used for generalization.

    OUTPUT: Vector layer
        Output layer that is copy of the input layer but with new
        attribute field "within_uoma" to indicate whether feature
        corresponds to natural watercourse.

    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    CLUSTER_RATIO = "CLUSTER_RATIO"

    def __init__(self) -> None:
        super().__init__()

        self._name = "detectwatercourseareasborder"
        self._display_name = "Detect Watercourse Areas Border"
        self._group_id = "utility"
        self._group = "Utility"
        self._short_help_string = ""

    def tr(self, string) -> str:
        """Returns a translatable string with the self.tr() function."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # noqa: N802
        return DetectWatercourseAreasBorder()

    def name(self) -> str:
        """Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return self._name

    def displayName(self) -> str:  # noqa: N802
        """Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr(self._display_name)

    def groupId(self) -> str:  # noqa: N802
        """Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return self._group_id

    def group(self) -> str:
        """Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr(self._group)

    def shortHelpString(self) -> str:  # noqa: N802
        """This algorithms tries to detect the watercourses forming the borders
        of dense watercourse areas (ojaverkosto, ojitetut suot etc). These
        borders should mostly be kept in the futher generalization.
        This algorithm consists of some basic steps:

        1. Form network faces of watercourse linestring layer ('the network')
        ('CartAGen:Calculate network faces'). The output of this step
        is a polygon layer ('Faces'). Non-connected part ('open') of the
        network gets removed in this step. The border of polygon features that
        are formed all coincide with some network feature of the input layer.
        2. Cluster the adjacent and nearby faces (features of the output in
        step 1) together using K-means clustering algorithm. The number of
        resulting clusters is given before the step, for example,
        Faces.featuresCount()*ratio_parameter? Output layer ('Clusters') has
        cluster id as an attribute.
        3. Features are dissolved by the 'CLUSTER_ID' attribute
        (native:dissolve). This step outputs a polygon layer (Dissolved)
        4. The border of the features in 'Dissolved' layer is extracted and
        outputted as a linestring layer ('OUTPUT').
        5. Compare output of step 4 with the input and if they overlap,
        tag the input feature field 'area_border' = True

        Parameters
        ----------
        INPUT: QgsVectorLayer
        CLUSTER_RATIO: integer
        OUTPUT: QgsVectorLayer

        """
        return self.tr(self._short_help_string)

    def initAlgorithm(self, configuration=None) -> None:  # noqa: N802
        """Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr("Input layer"),
                [QgsProcessing.SourceType.TypeVectorLine],
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr("WatercoursesWithAreaBorders"),
                QgsProcessing.SourceType.TypeVectorLine,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.CLUSTER_RATIO,
                self.tr("Approximate ratio of input features to clusters"),
                defaultValue=25,
            )
        )

    def processAlgorithm(  # noqa: N802
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Here is where the processing itself takes place."""
        # Initialize feedback if it is None
        if feedback is None:
            feedback = QgsProcessingFeedback()

        input_layer: QgsVectorLayer = self.parameterAsVectorLayer(
            parameters, self.INPUT, context
        )

        cluster_ratio: float = self.parameterAsDouble(
            parameters, self.CLUSTER_RATIO, context
        )

        # copy layer since we might have to delete some features and we don't
        # want to affect the actual data source

        copied_layer = input_layer.materialize(QgsFeatureRequest())
        input_fields = copied_layer.fields()
        output_fields = QgsFields()
        for field in input_fields:
            output_fields.append(field)
        output_fields.append(QgsField("area_border", QVariant.Bool))

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            output_fields,
            QgsWkbTypes.LineString,
            copied_layer.sourceCrs(),
        )

        feedback.setProgressText("Start processing!")
        feedback.setProgress(10)

        dropped_z_input = get_subalgorithm_result_layer(
            "native:dropmzvalues",
            {
                "INPUT": copied_layer,
                "DROP_M_VALUES": False,
                "DROP_Z_VALUES": True,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        feedback.setProgressText("Form network faces!")
        feedback.setProgress(20)

        network_faces = get_subalgorithm_result_layer(
            "CartAGen:Calculate network faces",
            {
                "INPUT": [dropped_z_input],
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        number_of_faces = network_faces.featureCount() or 0

        number_of_clusters = round(number_of_faces / cluster_ratio)

        feedback.setProgressText("Clustering faces!")
        feedback.setProgress(30)

        clusters = get_subalgorithm_result_layer(
            "native:kmeansclustering",
            {
                "INPUT": network_faces,
                "CLUSTERS": number_of_clusters,
                "FIELD_NAME": "CLUSTER_ID",
                "SIZE_FIELD_NAME": "CLUSTER_SIZE",
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        feedback.setProgressText("Dissolving features in clusters!")
        feedback.setProgress(40)

        dissolved = get_subalgorithm_result_layer(
            "native:dissolve",
            {
                "INPUT": clusters,
                "FIELD": "CLUSTER_ID",
                "SEPARATE_DISJOINT": False,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        feedback.setProgressText("Form the border lines of areas!")
        feedback.setProgress(50)

        borders = get_subalgorithm_result_layer(
            "qgis:convertgeometrytype",
            {
                "INPUT": dissolved,
                "TYPE": 2,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        total = (
            100.0 / copied_layer.featureCount() if copied_layer.featureCount() else 0
        )

        buffered_geoms = []
        for feature in borders.getFeatures():
            geom = feature.geometry()
            buffer = geom.buffer(1, segments=5)
            buffered_geoms.append(buffer)

        # Check for overlaps and update attribute
        copied_layer.startEditing()
        copied_layer.addAttribute(QgsField("area_border", QVariant.Bool))
        copied_layer.commitChanges()

        copied_layer.startEditing()
        for feature in copied_layer.getFeatures():
            geom = feature.geometry()
            # Check if it falls within any buffered geometry
            area_border = any(geom.within(buffer) for buffer in buffered_geoms)
            feature["area_border"] = area_border
            copied_layer.updateFeature(feature)

        copied_layer.commitChanges()

        for current, feature in enumerate(copied_layer.getFeatures()):
            if feedback.isCanceled():
                break

            sink.addFeature(QgsFeature(feature), QgsFeatureSink.Flag.FastInsert)

            feedback.setProgress(int(current * total))

        return {self.OUTPUT: dest_id}
