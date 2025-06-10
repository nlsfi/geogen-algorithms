from __future__ import annotations

from typing import Any

from qgis.core import (
    Qgis,
    QgsExpression,
    QgsFeatureIterator,
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsFeatureSource,
    QgsGeometry,
    QgsMapToPixelSimplifier,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsVectorLayer,
    QgsWkbTypes,
    edit,
)
from qgis.PyQt.QtCore import QCoreApplication
from yleistys_qgis_plugin.core.utils import (
    VISVALINGAM_ENUM_VERSION_CUTOFF,
    copy_feature_with_geometry,
    get_subalgorithm_result_layer,
)


class GeneralizeSeas(QgsProcessingAlgorithm):
    """This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    MIN_ISLAND_AREA = "MIN_ISLAND_AREA"
    MAX_SIMPLIFICATION_TOLERANCE = "MAX_SIMPLIFICATION_TOLERANCE"
    SIMPLIFY_TOLERANCE = "SIMPLIFY_TOLERANCE"
    EXAGGERATE_THIN_PARTS = "EXAGGERATE_THIN_PARTS"

    def __init__(self) -> None:
        super().__init__()

        self._name = "generalizeseas"
        self._display_name = "Generalize seas"
        self._group_id = "generalization"
        self._group = "Generalization"
        self._short_help_string = ""

    def tr(self, string) -> str:
        """Returns a translatable string with the self.tr() function."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # noqa: N802
        return GeneralizeSeas()

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
                self.SIMPLIFY_TOLERANCE,
                self.tr("Simplification tolerance"),
                defaultValue=10,
                minValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_ISLAND_AREA,
                self.tr("Island area threshold"),
                defaultValue=100,
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

        # used for comparing results
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.EXAGGERATE_THIN_PARTS,
                self.tr("Exaggerate thin parts"),
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
        min_island_area: int = self.parameterAsInt(
            parameters, self.MIN_ISLAND_AREA, context
        )
        max_simplification_tolerance: float = self.parameterAsDouble(
            parameters, self.MAX_SIMPLIFICATION_TOLERANCE, context
        )
        simplify_tolerance: int = self.parameterAsInt(
            parameters, self.SIMPLIFY_TOLERANCE, context
        )
        exaggerate_thin_parts: bool = self.parameterAsBoolean(
            parameters, self.EXAGGERATE_THIN_PARTS, context
        )

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            input_layer.fields(),
            input_layer.wkbType(),
            input_layer.sourceCrs(),
        )

        if feedback is None:
            feedback = QgsProcessingFeedback()

        # There's multiple stages where dissolving is required,
        # save for convenience and clarity
        def _dissolve(layer: QgsVectorLayer) -> QgsVectorLayer:
            params = {
                "INPUT": layer,
                "SEPARATE_DISJOINT": False,
                "OUTPUT": "TEMPORARY_OUTPUT",
            }

            return get_subalgorithm_result_layer(
                "native:dissolve", params, context, feedback
            )

        # There's multiple stages where the multipart to single part algorithm
        # is required, save for convenience and clarity
        def _multiparttosingleparts(layer: QgsVectorLayer) -> QgsVectorLayer:
            params = {
                "INPUT": layer,
                "OUTPUT": "TEMPORARY_OUTPUT",
            }

            return get_subalgorithm_result_layer(
                "native:multiparttosingleparts", params, context, feedback
            )

        # Part of the process is to simplify and smooth the polygons.
        # The sea features also include the territorial water borders.
        # These features are large along with having a small number
        # of vertices which causes the smoothing operation to distort
        # the borders significantly.
        # For now handle this issue by extracting the relevant features
        # and combine them with the result at the end.
        expr = QgsExpression("territorial_waters_category_id > 1")
        categories_dissolved = _dissolve(
            input_layer.materialize(QgsFeatureRequest(expr))
        )
        territorial_exterior_ring = get_subalgorithm_result_layer(
            "native:deleteholes",
            {
                "INPUT": categories_dissolved,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        # Processing features one-by-one or in clusters introduces
        # many issues, such as having to deal with islands which
        # are not holes but instead between multiple features and the
        # fact that exaggerating thin parts may modify the features'
        # geometries and topology such that it'll create new holes
        # between the features.

        # Dissolving all adjacent features to one large feature will
        # make processing significantly slower, but it solves most of
        # these issues
        layer = _dissolve(input_layer)
        layer = _multiparttosingleparts(layer)

        feedback.setProgressText("Eliminating features")

        # Remove small islands first
        eliminated_layer = get_subalgorithm_result_layer(
            "native:deleteholes",
            {
                "INPUT": layer,
                "MIN_AREA": min_island_area,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        # Exaggerate thin parts. The islands are added back in
        # as interior rings later and this operation may change
        # the geometry of an island so this has to be done before
        # extracting the islands, otherwise its effect would be
        # nullified.
        if exaggerate_thin_parts:
            feedback.setProgressText("Exaggerating thin parts")

            eliminated_layer = get_subalgorithm_result_layer(
                "yleistys_qgis_plugin:exaggeratethinparts",
                {
                    "INPUT": eliminated_layer,
                    "WIDTH_TOLERANCE": 20,
                    "EXAGGERATE_BY": 3,
                    "MAX_THINNESS": 0.5,
                    "MIN_AREA": 200,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context,
                feedback,
            )

        # The exterior ring and islands need to be processed differently,
        # f.e. small islands should't be simplified as much, so extract
        # all islands at this point and process them separately
        extracted_islands = get_subalgorithm_result_layer(
            "yleistys_qgis_plugin:extractinteriorrings",
            {
                "INPUT": eliminated_layer,
                "FEATURE_PER_RING": True,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        generalized_islands = get_subalgorithm_result_layer(
            "yleistys_qgis_plugin:generalizeislands",
            {
                "INPUT": extracted_islands,
                "MIN_WIDTH": 185,
                "MIN_ELONGATION": 3.349,
                "MAX_SIMPLIFICATION_TOLERANCE": max_simplification_tolerance,
                "EXAGGERATE_BY": 3,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        # Delete the islands to process the exterior ring separately
        exterior_ring = get_subalgorithm_result_layer(
            "native:deleteholes",
            {
                "INPUT": eliminated_layer,
                "MIN_AREA": 0.0,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        if Qgis.QGIS_VERSION_INT < VISVALINGAM_ENUM_VERSION_CUTOFF:  # pyright: ignore
            simplifier = QgsMapToPixelSimplifier(
                QgsMapToPixelSimplifier.SimplifyFlag.SimplifyGeometry,
                simplify_tolerance,
                QgsMapToPixelSimplifier.SimplifyAlgorithm.Visvalingam,
            )
        else:
            simplifier = QgsMapToPixelSimplifier(
                QgsMapToPixelSimplifier.SimplifyFlag.SimplifyGeometry,
                simplify_tolerance,
                Qgis.VectorSimplificationAlgorithm.Visvalingam,  # pyright: ignore
            )

        feedback.setProgressText("Simplifying and smoothing")

        total = (
            100.0 / exterior_ring.featureCount() if exterior_ring.featureCount() else 0
        )

        # Perform simplification and smoothing feature-by-feature
        with edit(exterior_ring):
            features: QgsFeatureIterator = exterior_ring.getFeatures()
            for current, feature in enumerate(features):
                if feedback.isCanceled():
                    break

                geom = feature.geometry()

                # simplify geometries with more vertices a greater amount,
                # capping at max_simplification_tolerance
                vertex_count: int = geom.constGet().nCoordinates()
                add_to_tolerance: float = vertex_count / 200
                add_to_tolerance = min(
                    add_to_tolerance, max_simplification_tolerance - 1
                )

                simplifier.setTolerance(1 + add_to_tolerance)

                new_geom = simplifier.simplify(geom)

                # simplification may introduce duplicate nodes, delete
                new_geom.removeDuplicateNodes()

                new_geom = new_geom.smooth()

                exterior_ring.changeGeometry(feature.id(), new_geom)

                feedback.setProgress(int(current * total))

        feedback.setProgressText("Adding generalized islands")

        generalized_sea = get_subalgorithm_result_layer(
            "native:difference",
            {
                "INPUT": exterior_ring,
                "OVERLAY": generalized_islands,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        diff = get_subalgorithm_result_layer(
            "native:difference",
            {
                "INPUT": generalized_sea,
                "OVERLAY": territorial_exterior_ring,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        diff_singles = _multiparttosingleparts(diff)

        biggest_geom: QgsGeometry = QgsGeometry()
        for feature in diff_singles.getFeatures():
            geom = feature.geometry()
            if geom.area() > biggest_geom.area():
                biggest_geom = geom

        combined = territorial_exterior_ring.getGeometry(1).combine(biggest_geom)

        with edit(generalized_sea):
            res = generalized_sea.changeGeometry(1, combined)
            feedback.pushInfo(f"result: {res}")

        # the difference algorithm changes the geometry type to
        # multipolygon, convert back to single polygon
        if QgsWkbTypes.isSingleType(input_layer.wkbType()):
            generalized_sea = _multiparttosingleparts(generalized_sea)

        for gen_feature in generalized_sea.getFeatures():
            if feedback.isCanceled():
                break

            geom = gen_feature.geometry()
            sink.addFeature(
                copy_feature_with_geometry(gen_feature, geom),
                QgsFeatureSink.Flag.FastInsert,
            )

        return {self.OUTPUT: dest_id}
