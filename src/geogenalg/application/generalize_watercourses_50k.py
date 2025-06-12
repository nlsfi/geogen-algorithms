from __future__ import annotations

from typing import Any

from qgis.core import (
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsFeatureSource,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
)
from qgis.PyQt.QtCore import QCoreApplication, QVariant
from yleistys_qgis_plugin.core.utils import (
    get_subalgorithm_result_layer,
)


class GeneralizeWatercourses50K(QgsProcessingAlgorithm):
    """ """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    # A vector layer that contains natural waterways for reference
    UOMA_LAYER = "UOMA_LAYER"
    USE_UOMA = "USE_UOMA"

    def __init__(self) -> None:
        super().__init__()

        self._name = "generalizewatercourses50k"
        self._display_name = "Generalize watercourses 50K"
        self._group_id = "generalization"
        self._group = "Generalization"
        self._short_help_string = ""

    def tr(self, string) -> str:
        """Returns a translatable string with the self.tr() function."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # noqa: N802
        return GeneralizeWatercourses50K()

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
            QgsProcessingParameterFeatureSource(
                self.UOMA_LAYER,
                self.tr("Input reference natural waterway layer (uoma)"),
                [QgsProcessing.SourceType.TypeVectorLine],
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_UOMA,
                self.tr("Use uoma layer to detect natural waterways"),
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr("Generalized"))
        )

    def processAlgorithm(  # noqa: N802
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Here is where the processing itself takes place."""
        input_layer: QgsFeatureSource = self.parameterAsVectorLayer(
            parameters, self.INPUT, context
        )

        uoma_layer: QgsFeatureSource = self.parameterAsVectorLayer(
            parameters, self.UOMA_LAYER, context
        )

        use_uoma: bool = self.parameterAsBoolean(parameters, self.USE_UOMA, context)

        copied_layer = input_layer.materialize(QgsFeatureRequest())
        copied_layer.startEditing()
        copied_layer.addAttribute(QgsField("too_short", QVariant.Bool))
        copied_layer.addAttribute(QgsField("remove", QVariant.Bool))
        copied_layer.addAttribute(QgsField("length", QVariant.Double))
        copied_layer.addAttribute(QgsField("connected", QVariant.Bool))
        copied_layer.addAttribute(QgsField("overlap_uoma", QVariant.Bool))
        copied_layer.addAttribute(QgsField("area_border", QVariant.Bool))
        copied_layer.commitChanges()

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            copied_layer.fields(),
            copied_layer.wkbType(),
            copied_layer.sourceCrs(),
        )

        if feedback is None:
            feedback = QgsProcessingFeedback()

        cleaned_input = get_subalgorithm_result_layer(
            "yleistys_qgis_plugin:removeshortwatercourses",
            {
                "INPUT": input_layer,
                "MIN_LENGTH": 50,
                "BUFFER_RADIUS": 15,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        if use_uoma:
            feedback.setProgressText(
                "Identifying natural watercourses in the input layer"
            )

            cleaned_input = get_subalgorithm_result_layer(
                "yleistys_qgis_plugin:detectnaturawatercourseslwithuomaoverlap",
                {
                    "INPUT": cleaned_input,
                    "INPUT_UOMA": uoma_layer,
                    "BUFFER": 30,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context,
                feedback,
            )

        borders_identified = get_subalgorithm_result_layer(
            "yleistys_qgis_plugin:detectwatercourseareasborder",
            {
                "INPUT": cleaned_input,
                "OUTPUT": "TEMPORARY_OUTPUT",
                "CLUSTER_RATIO": 25,
            },
            context,
            feedback,
        )

        dense_lines_removed = get_subalgorithm_result_layer(
            "yleistys_qgis_plugin:removedensewatercourses",
            {
                "INPUT": borders_identified,
                "DISTANCE_THRESHOLD": 50,
                "LINE_DENSITY_RATIO": 20,
                "STROKES": False,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        results_layer = dense_lines_removed

        feedback.setProgressText("Simplifying and smoothing")

        total = (
            100.0 / results_layer.featureCount() if results_layer.featureCount() else 0
        )

        for current, feature in enumerate(borders_identified.getFeatures()):
            if feedback.isCanceled():
                break

            sink.addFeature(feature, QgsFeatureSink.Flag.FastInsert)

            feedback.setProgress(int(current * total))

        return {self.OUTPUT: dest_id}
