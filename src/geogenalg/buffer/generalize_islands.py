from __future__ import annotations

from typing import Any

from qgis.core import (
    Qgis,
    QgsFeature,
    QgsFeatureSink,
    QgsFeatureSource,
    QgsMapToPixelSimplifier,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QCoreApplication
from yleistys_qgis_plugin.core.utils import (
    VISVALINGAM_ENUM_VERSION_CUTOFF,
    copy_feature_with_geometry,
)


class GeneralizeIslands(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    MIN_WIDTH = "MIN_WIDTH"
    MIN_ELONGATION = "MIN_ELONGATION"
    MAX_SIMPLIFICATION_TOLERANCE = "MAX_SIMPLIFICATION_TOLERANCE"
    EXAGGERATE_BY = "EXAGGERATE_BY"

    def __init__(self) -> None:
        super().__init__()

        self._name = "generalizeislands"
        self._display_name = "Generalize islands"
        self._group_id = "utility"
        self._group = "Utility"
        self._short_help_string = ""

    def tr(self, string) -> str:
        """Returns a translatable string with the self.tr() function."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # noqa: N802
        return GeneralizeIslands()

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
                [QgsProcessing.SourceType.TypeVectorPolygon],
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_WIDTH,
                self.tr("Exaggerate only if max width under"),
                defaultValue=185,
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_ELONGATION,
                self.tr("Exaggerate only if elongation over"),
                defaultValue=3.349,
                type=QgsProcessingParameterNumber.Type.Double,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.EXAGGERATE_BY,
                self.tr("Exaggerate by"),
                defaultValue=3,
                minValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_SIMPLIFICATION_TOLERANCE,
                self.tr("Maximum simplification tolerance"),
                defaultValue=10,
                minValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT, self.tr("Generalized islands")
            )
        )

    def processAlgorithm(  # noqa: N802
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Here is where the processing itself takes place."""
        if feedback is None:
            feedback = QgsProcessingFeedback()

        source: QgsFeatureSource = self.parameterAsSource(
            parameters, self.INPUT, context
        )
        min_width: float = self.parameterAsDouble(parameters, self.MIN_WIDTH, context)
        min_elongation: float = self.parameterAsDouble(
            parameters, self.MIN_ELONGATION, context
        )
        max_simplification_tolerance: float = self.parameterAsDouble(
            parameters, self.MAX_SIMPLIFICATION_TOLERANCE, context
        )
        exaggerate_by: float = self.parameterAsDouble(
            parameters, self.EXAGGERATE_BY, context
        )

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            source.fields(),
            source.wkbType(),
            source.sourceCrs(),
        )

        total = 100.0 / source.featureCount() if source.featureCount() else 0

        if Qgis.QGIS_VERSION_INT < VISVALINGAM_ENUM_VERSION_CUTOFF:  # pyright: ignore
            simplifier = QgsMapToPixelSimplifier(
                QgsMapToPixelSimplifier.SimplifyFlag.SimplifyGeometry,
                1,
                QgsMapToPixelSimplifier.SimplifyAlgorithm.Visvalingam,
            )
        else:
            simplifier = QgsMapToPixelSimplifier(
                QgsMapToPixelSimplifier.SimplifyFlag.SimplifyGeometry,
                1,
                Qgis.VectorSimplificationAlgorithm.Visvalingam,  # pyright: ignore
            )

        feature: QgsFeature
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom = feature.geometry()

            _, _, _, ombb_width, ombb_height = geom.orientedMinimumBoundingBox()
            elongation: float = ombb_height / ombb_width

            # check whether the island should be exaggerated
            if elongation >= min_elongation and ombb_width < min_width:
                geom = geom.buffer(distance=exaggerate_by, segments=2)

            vertex_count: int = geom.constGet().nCoordinates()
            add_to_tolerance: float = vertex_count / 200
            add_to_tolerance = min(add_to_tolerance, max_simplification_tolerance - 1)

            simplifier.setTolerance(1 + add_to_tolerance)

            simplified_geom = simplifier.simplify(geom)
            smoothed_geom = simplified_geom.smooth()
            sink.addFeature(
                copy_feature_with_geometry(feature, smoothed_geom),
                QgsFeatureSink.Flag.FastInsert,
            )

            feedback.setProgress(current * total)

        return {self.OUTPUT: dest_id}
