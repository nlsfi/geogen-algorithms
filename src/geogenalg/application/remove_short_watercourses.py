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

from yleistys_qgis_plugin.core.watercourse_structure_detection import check_line_connectivity


class RemoveShortWatercourses(QgsProcessingAlgorithm):
    """This algorithm cleans the watercourse input layer by
    removing short linestrings that are not connected to
    the rest of the watercourse network. The line feature
    is removed from the layer if it is shorter than threshold
    parameter and both of the ends are not connected to some
    other line string. This latter condition is checked by
    forming buffer on each end and detecting if there is any
    intersection with the buffer and some other feature.

    Parameters
    ----------
    INPUT: Input vector linestring layer.
    OUTPUT: Vector layer of linestrings where the short and
    non-connected features are removed.
    MIN_LENGTH: Float number.
        Only line features that are shorter than this value
        are considered for removal.
    BUFFER_RADIUS: Float number.
        The radius for the buffers that are constructed on
        each end vertex.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    MIN_LENGTH = "MIN_LENGTH"
    BUFFER_RADIUS = "BUFFER_RADIUS"

    def __init__(self) -> None:
        super().__init__()

        self._name = "removeshortwatercourses"
        self._display_name = "Remove Short Watercourses"
        self._group_id = "utility"
        self._group = "Utility"
        self._short_help_string = """Remove short linestrings that are
                                    not connected to other lines, or are
                                    connected only by one end"""

    def tr(self, string) -> str:
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # noqa N802
        return RemoveShortWatercourses()

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
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and
        the parameters and outputs associated with it..
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
            QgsProcessingParameterNumber(
                self.MIN_LENGTH,
                self.tr("Minimum length for the lines too keep"),
                defaultValue=50,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUFFER_RADIUS,
                self.tr(
                    """How far away line are considered as non-connected from
                    the others"""
                ),
                defaultValue=15,
            )
        )

        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr("ShortWatercoursesRemoved")))

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

        input_layer: QgsVectorLayer = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        threshold_length: float = self.parameterAsDouble(parameters, self.MIN_LENGTH, context)

        buffer_radius: float = self.parameterAsDouble(parameters, self.BUFFER_RADIUS, context)

        # copy layer since we might have to delete some features and we don't
        # want to affect the actual data source
        copied_layer = input_layer.materialize(QgsFeatureRequest())
        copied_layer.startEditing()
        copied_layer.addAttribute(QgsField("too_short", QVariant.Bool))
        copied_layer.addAttribute(QgsField("remove", QVariant.Bool))
        copied_layer.addAttribute(QgsField("length", QVariant.Double))
        copied_layer.addAttribute(QgsField("connected", QVariant.Bool))
        copied_layer.commitChanges()

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            copied_layer.fields(),
            copied_layer.wkbType(),
            copied_layer.sourceCrs(),
        )

        feedback.setProgress(0)
        feedback.setProgressText("Check the length & connectivity of lines")

        total = 100.0 / copied_layer.featureCount() if copied_layer.featureCount() else 0

        for current, feature in enumerate(copied_layer.getFeatures()):
            if feedback.isCanceled():
                break

            new_feature = feature

            feature.setFields(copied_layer.fields(), False)
            new_feature.resizeAttributes(new_feature.fields().size())

            feature["length"] = feature.geometry().length()
            if feature.geometry().length() < threshold_length:
                feature["too_short"] = True
            else:
                feature["too_short"] = False

            feature["connected"] = check_line_connectivity(feature.geometry(), copied_layer, buffer_radius)

            if feature["too_short"] and not feature["connected"]:
                feature["remove"] = True
            else:
                feature["remove"] = False

            if not feature["remove"]:
                sink.addFeature(QgsFeature(feature), QgsFeatureSink.Flag.FastInsert)

            feedback.setProgress(int(current * total))

        return {self.OUTPUT: dest_id}
