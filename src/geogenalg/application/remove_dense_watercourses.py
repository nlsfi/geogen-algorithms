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
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import (
    QCoreApplication,
)
from yleistys_qgis_plugin.core.utils import get_subalgorithm_result_layer


class RemoveDenseWatercourses(QgsProcessingAlgorithm):
    """ """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    LINE_DENSITY_RATIO = "LINE_DENSITY_RATIO"
    DISTANCE_THRESHOLD = "DISTANCE_THRESHOLD"
    STROKES = "STROKES"

    def __init__(self) -> None:
        super().__init__()

        self._name = "removedensewatercourses"
        self._display_name = "Remove Dense Watercourses"
        self._group_id = "utility"
        self._group = "Utility"
        self._short_help_string = ""

    def tr(self, string) -> str:
        """Returns a translatable string with the self.tr() function."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # noqa: N802
        return RemoveDenseWatercourses()

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
        """Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
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
            QgsProcessingParameterNumber(
                self.DISTANCE_THRESHOLD,
                self.tr(
                    "Maximum distance for clustering parallel lines (in CRS units)"
                ),
                defaultValue=50,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.LINE_DENSITY_RATIO,
                self.tr("Percentage of lines to keep withing each cluster"),
                defaultValue=50,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.STROKES,
                self.tr(
                    "Form river strokes from input geometries first? (Option not implemented yet)"
                ),
                defaultValue=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT, self.tr("RemoveDenseWatercoursesOutput")
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

        distance_threshold: float = self.parameterAsDouble(
            parameters, self.DISTANCE_THRESHOLD, context
        )

        density_ratio: float = self.parameterAsDouble(
            parameters, self.LINE_DENSITY_RATIO, context
        )

        form_strokes: bool = self.parameterAsBool(parameters, self.STROKES, context)
        # copy layer since we might have to delete some features and we don't
        # want to affect the actual data source

        copied_layer = input_layer.materialize(QgsFeatureRequest())

        input_fields = copied_layer.fields()
        """
        output_fields = QgsFields()
        for field in input_fields:
            output_fields.append(field)
        output_fields.append(QgsField("area_border", QVariant.Bool))"""

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            input_fields,
            copied_layer.wkbType(),
            copied_layer.sourceCrs(),
        )

        feedback.setProgressText("Start processing!")
        feedback.setProgress(10)

        centroids = get_subalgorithm_result_layer(
            "native:centroids",
            {
                "INPUT": copied_layer,
                "ALL_PARTS": False,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        clusters = get_subalgorithm_result_layer(
            "native:dbscanclustering",
            {
                "INPUT": centroids,
                "MIN_SIZE": 1,
                "EPS": distance_threshold,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        buffered_clusters = get_subalgorithm_result_layer(
            "native:buffer",
            {
                "INPUT": clusters,
                "DISTANCE": 6,
                "SEGMENTS": 5,
                "END_CAP_STYLE": 0,
                "JOIN_STYLE": 0,
                "MITER_LIMIT": 2,
                "DISSOLVE": False,
                "SEPARATE_DISJOINT": False,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        joined_layer = get_subalgorithm_result_layer(
            "native:joinattributesbylocation",
            {
                "INPUT": copied_layer,
                "PREDICATE": [0, 3],
                "JOIN": buffered_clusters,
                "JOIN_FIELDS": ["CLUSTER_ID"],
                "METHOD": 0,
                "DISCARD_NONMATCHING": False,
                "PREFIX": "",
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        remove_dense_lines = joined_layer = get_subalgorithm_result_layer(
            "qgis:randomextractwithinsubsets",
            {
                "INPUT": joined_layer,
                "FIELD": "CLUSTER_ID",
                "METHOD": 1,
                "NUMBER": density_ratio,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        # processed_qgis_layer = joined_layer
        processed_qgis_layer = remove_dense_lines

        total = (
            100.0 / copied_layer.featureCount() if copied_layer.featureCount() else 0
        )

        for current, feature in enumerate(processed_qgis_layer.getFeatures()):
            if feedback.isCanceled():
                break

            sink.addFeature(QgsFeature(feature), QgsFeatureSink.Flag.FastInsert)

            feedback.setProgress(int(current * total))

        return {self.OUTPUT: dest_id}
