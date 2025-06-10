from __future__ import annotations

from typing import Any

from qgis.core import (
    Qgis,
    QgsFeature,
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsFeatureSource,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsRectangle,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QCoreApplication
from yleistys_qgis_plugin.core.utils import (
    extract_interior_rings,
)


class ExtractInteriorRings(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    def __init__(self) -> None:
        super().__init__()

        self._name = "extractinteriorrings"
        self._display_name = "Extract interior rings (holes) from layer"
        self._group_id = "utility"
        self._group = "Utility"
        self._short_help_string = ""

    def tr(self, string) -> str:
        """Returns a translatable string with the self.tr() function."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # noqa: N802
        return ExtractInteriorRings()

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
            QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr("Extracted"))
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

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            source.fields(),
            source.wkbType(),
            source.sourceCrs(),
        )

        geom_rings: list[QgsGeometry] = []

        for feature in source.getFeatures():
            if feedback.isCanceled():
                return {}
            geom_rings.append(extract_interior_rings(feature.geometry()))

        for geom_ring in geom_rings:
            if feedback.isCanceled():
                return {}
            for primitive_ring in geom_ring.constParts():
                if feedback.isCanceled():
                    return {}

                interior_ring: QgsGeometry = QgsGeometry(primitive_ring.clone())

                # let's check if there are any other exterior rings inside the interior
                # ring and add those as rings to handle recursive islands
                rect: QgsRectangle = interior_ring.boundingBox()

                req = QgsFeatureRequest().setFilterRect(rect)

                for feat in source.getFeatures(req):
                    geom = feat.geometry()
                    if geom.within(interior_ring):
                        rings: list[list[QgsPointXY]]

                        if QgsWkbTypes.isSingleType(geom.wkbType()):
                            rings = [geom.asPolygon()[0]]
                        else:
                            rings = [ring[0] for ring in geom.asMultiPolygon()]

                        for ring in rings:
                            res = interior_ring.addRing(ring)
                            if res != Qgis.GeometryOperationResult.Success:
                                feedback.reportError(
                                    "Could not add ring to extracted interior ring!",
                                    fatalError=True,
                                )

                feature = QgsFeature(source.fields())
                feature.setGeometry(interior_ring)
                sink.addFeature(feature, QgsFeatureSink.Flag.FastInsert)

        return {self.OUTPUT: dest_id}
