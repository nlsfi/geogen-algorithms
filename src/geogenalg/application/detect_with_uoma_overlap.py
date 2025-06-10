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
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QVariant,
)


class DetectWithUomaOverlap(QgsProcessingAlgorithm):
    """
    The algorithm detects those features of the input layer
    that lie within the buffer formed around the reference layer
    (uoma layer)

    Parameters
    -----------
    INPUT: Vector layer
        The input layer that is used for generalization.
    INPUT_UOMA: Vector layer
        Reference layer that contains natural watercourse
        features that we do not want to generalize away.
    BUFFER: An integer number
        The buffer width that acts as a tolerance threshold since
        the corresponding input and reference layer feature geometries
        may not be exactly equal.
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
    BUFFER = "BUFFER"
    INPUT_UOMA = "INPUT_UOMA"

    def __init__(self) -> None:
        super().__init__()

        self._name = "detectnaturawatercourseslwithuomaoverlap"
        self._display_name = "Detect Natural Watercourses With Uoma Overlap"
        self._group_id = "utility"
        self._group = "Utility"
        self._short_help_string = """
        DetectWithUomaOverlap algorithm takes an input vector layer of
        LineStrings as the layer that should be generalized. Another input
        vector layer (Uoma) acts as a reference layer for the natural waterways
        as linestrings. This algorithm tries to detect those features in the
        input layer that correspond to natural waterways. Since the
        corresponding geometries do not exactly match, a buffer for Uoma
        geometries is used. The buffer parameter should be large enough so that
        all the relevant 'natural' watercourse geometries of the input layer
        are contained fully within them.
        """

    def tr(self, string) -> str:
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # noqa N802
        return DetectWithUomaOverlap()

    def name(self) -> str:
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return self._name

    def displayName(self) -> str:  # noqa N802
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr(self._display_name)

    def groupId(self) -> str:  # noqa N802
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return self._group_id

    def group(self) -> str:
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr(self._group)

    def shortHelpString(self) -> str:  # noqa N802
        """
        DetectWithUomaOverlap algorithm takes an input vector layer of
        LineStrings as the layer that should be generalized. Another input
        vector layer (Uoma) acts as a reference layer for the natural waterways
        as linestrings. This algorithm tries to detect those features in the
        input layer that correspond to natural waterways. Since the
        corresponding geometries do not exactly match, a buffer for Uoma
        geometries is used. The buffer parameter should be large enough so that
        all the relevant 'natural' watercourse geometries are of input layer
        are contained within them.
        """
        return self.tr(self._short_help_string)

    def initAlgorithm(self, configuration=None):  # noqa N802
        """
        Here we define the inputs and output of the algorithm, along
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
                self.INPUT_UOMA,
                self.tr("Input Uoma layer"),
                [QgsProcessing.SourceType.TypeVectorLine],
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUFFER,
                self.tr("Buffer width for Uoma layer"),
                defaultValue=30,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr("DetectNaturalWithUoma")))

    def processAlgorithm(  # noqa N802
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """
        Here is where the processing itself takes place.
        """

        # Initialize feedback if it is None
        if feedback is None:
            feedback = QgsProcessingFeedback()

        input_layer: QgsVectorLayer = self.parameterAsVectorLayer(parameters, self.INPUT, context)  # input_layer
        reference_layer: QgsVectorLayer = self.parameterAsVectorLayer(
            parameters, self.INPUT_UOMA, context
        )  # input_uoma_layer

        # buffer_radius
        tolerance: int = self.parameterAsInt(parameters, self.BUFFER, context)
        # copy layer since we might have to delete some features and we don't
        # want to affect the actual data source

        copied_layer = input_layer.materialize(QgsFeatureRequest())
        copied_layer.startEditing()
        copied_layer.addAttribute(QgsField("overlap_uoma", QVariant.Bool))
        copied_layer.commitChanges()

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            copied_layer.fields(),
            copied_layer.wkbType(),
            copied_layer.sourceCrs(),
        )

        feedback.setProgressText("Start processing!")
        feedback.setProgress(10)

        total = 100.0 / copied_layer.featureCount() if copied_layer.featureCount() else 0
        # Collect buffered geometries from reference_layer
        buffered_geoms = []
        for feature in reference_layer.getFeatures():
            geom = feature.geometry()
            buffer = geom.buffer(tolerance, segments=5)
            buffered_geoms.append(buffer)

        # Check for overlaps and update attribute
        # with edit(copied_layer):
        copied_layer.startEditing()

        for feature in copied_layer.getFeatures():
            geom = feature.geometry()
            # Check if it falls within any buffered geometry
            is_within = any(geom.within(buffer) for buffer in buffered_geoms)
            feature["overlap_uoma"] = is_within
            copied_layer.updateFeature(feature)

        copied_layer.commitChanges()

        for current, feature in enumerate(copied_layer.getFeatures()):
            if feedback.isCanceled():
                break

            sink.addFeature(QgsFeature(feature), QgsFeatureSink.Flag.FastInsert)

            feedback.setProgress(int(current * total))

        return {self.OUTPUT: dest_id}
