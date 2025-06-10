from __future__ import annotations

from typing import Any

from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsFeatureSource,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QCoreApplication
from yleistys_qgis_plugin.core.exaggeration import buffer_elimination, filter_and_buffer
from yleistys_qgis_plugin.core.utils import copy_feature_with_geometry


class ExaggerateThinParts(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    WIDTH_TOLERANCE = "WIDTH_TOLERANCE"
    MAX_THINNESS = "MAX_THINNESS"
    MIN_AREA = "MIN_AREA"
    EXAGGERATE_BY = "EXAGGERATE_BY"
    COMBINE = "COMBINE"

    def __init__(self) -> None:
        super().__init__()

        self._name = "exaggeratethinparts"
        self._display_name = "Exaggerate thin parts"
        self._group_id = "utility"
        self._group = "Utility"
        self._short_help_string = ""

    def tr(self, string) -> str:
        """Returns a translatable string with the self.tr() function."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # noqa: N802
        return ExaggerateThinParts()

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
                self.WIDTH_TOLERANCE,
                self.tr("Width tolerance"),
                defaultValue=10,
                minValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.EXAGGERATE_BY,
                self.tr("Exaggerate by"),
                defaultValue=10,
                minValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_THINNESS,
                self.tr("Maximum thinness"),
                defaultValue=0.5,
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                maxValue=1.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_AREA,
                self.tr("Minimum area"),
                defaultValue=200,
                type=QgsProcessingParameterNumber.Type.Double,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMBINE,
                self.tr("Combine into geometries"),
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr("Exaggerated"))
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
        width_tolerance = self.parameterAsInt(parameters, self.WIDTH_TOLERANCE, context)
        exaggerate_by = self.parameterAsInt(parameters, self.EXAGGERATE_BY, context)
        max_thinness = self.parameterAsInt(parameters, self.MAX_THINNESS, context)
        min_area = self.parameterAsInt(parameters, self.MIN_AREA, context)
        combine = self.parameterAsBool(parameters, self.COMBINE, context)

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            source.fields(),
            source.wkbType(),
            source.sourceCrs(),
        )

        total = 100.0 / source.featureCount() if source.featureCount() else 0

        feature: QgsFeature
        for current, feature in enumerate(source.getFeatures()):
            geom = feature.geometry()

            eliminated_buffer = buffer_elimination(geom, width_tolerance / 2)
            diff = geom.difference(eliminated_buffer)

            exaggerated_areas: QgsGeometry | None = filter_and_buffer(
                diff, exaggerate_by, min_area, max_thinness
            )

            feedback.setProgress(int(current * total))

            if combine:
                final_geom = (
                    geom.combine(exaggerated_areas) if exaggerated_areas else geom
                )
            else:
                final_geom = exaggerated_areas or None

                if not final_geom:
                    continue

            sink.addFeature(
                copy_feature_with_geometry(feature, final_geom),
                QgsFeatureSink.Flag.FastInsert,
            )

        return {self.OUTPUT: dest_id}
