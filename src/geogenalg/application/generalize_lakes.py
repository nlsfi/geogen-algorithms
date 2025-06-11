from __future__ import annotations

from yleistys_qgis_plugin.core.utils import (
    VISVALINGAM_ENUM_VERSION_CUTOFF,
    extract_interior_rings_from_layer,
    get_subalgorithm_result_layer,
)

try:
    import processing
except ImportError:
    from qgis import processing

from typing import Any

from qgis.core import (
    Qgis,
    QgsExpression,
    QgsFeature,
    QgsFeatureIterator,
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsMapToPixelSimplifier,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeatureSourceDefinition,
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


class GeneralizeLakes(QgsProcessingAlgorithm):
    """This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsGeneralizeLakes
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    MIN_LAKE_AREA = "MIN_LAKE_AREA"
    MIN_ISLAND_AREA = "MIN_ISLAND_AREA"
    MAX_SIMPLIFICATION_TOLERANCE = "MAX_SIMPLIFICATION_TOLERANCE"
    EXAGGERATE_THIN_PARTS = "EXAGGERATE_THIN_PARTS"

    def __init__(self) -> None:
        super().__init__()

        self._name = "generalizelakes"
        self._display_name = "Generalize lakes"
        self._group_id = "generalization"
        self._group = "Generalization"
        self._short_help_string = ""

    def tr(self, string) -> str:
        """Returns a translatable string with the self.tr() function."""
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):  # noqa: N802
        return GeneralizeLakes()

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
                self.MIN_LAKE_AREA,
                self.tr("Remove lakes with an area under"),
                defaultValue=0,
                minValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_ISLAND_AREA,
                self.tr("Remove islands with an area under"),
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
        # Initialize feedback if it is None
        if feedback is None:
            feedback = QgsProcessingFeedback()

        def check_feature_count_difference(
            layer_1: QgsVectorLayer, layer_2: QgsVectorLayer, *, warn_if_different: bool
        ) -> None:
            difference = layer_1.featureCount() - layer_2.featureCount()

            log = (
                feedback.pushWarning
                if warn_if_different and difference != 0
                else feedback.pushInfo
            )

            log(
                f"Difference in feature count is: {difference} ({layer_1.featureCount()}/{layer_2.featureCount()})"
            )

        input_layer: QgsVectorLayer = self.parameterAsVectorLayer(
            parameters, self.INPUT, context
        )
        min_lake_area: int = self.parameterAsInt(
            parameters, self.MIN_LAKE_AREA, context
        )
        min_island_area: int = self.parameterAsInt(
            parameters, self.MIN_ISLAND_AREA, context
        )
        max_simplification_tolerance: float = self.parameterAsDouble(
            parameters, self.MAX_SIMPLIFICATION_TOLERANCE, context
        )
        exaggerate_thin_parts: bool = self.parameterAsBoolean(
            parameters, self.EXAGGERATE_THIN_PARTS, context
        )

        # copy layer since we might have to delete some features and we don't want
        # to affect the actual data source
        copied_layer = input_layer.materialize(QgsFeatureRequest())

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            copied_layer.fields(),
            copied_layer.wkbType(),
            copied_layer.sourceCrs(),
        )

        feedback.setProgress(0)
        feedback.setProgressText("Selecting features to merge")

        # Select all features that touch another feature with processing script
        processing.run(  # pyright: ignore
            "native:selectbylocation",
            {
                "INPUT": input_layer,
                "PREDICATE": [4, 5],  # touch or overlap
                "INTERSECT": input_layer,  # layer to compare features to
                "METHOD": 0,  # create a new selection
            },
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )

        feedback.pushInfo(
            f"Selected {input_layer.selectedFeatureCount()}/{input_layer.featureCount()} features to be merged."
        )

        feedback.setProgressText("Merging features")

        # Looks like we have to use the input layer here because using the
        # temporary copied layer causes problems, or would have to be
        # added to the project layer tree which is not wanted

        if input_layer.selectedFeatureCount() > 0:
            dissolve_layer = get_subalgorithm_result_layer(
                "native:dissolve",
                {
                    "INPUT": QgsProcessingFeatureSourceDefinition(
                        input_layer.source(),
                        selectedFeaturesOnly=True,
                    ),
                    "SEPARATE_DISJOINT": False,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context,
                feedback,
            )

            combined_features_layer = get_subalgorithm_result_layer(
                "native:multiparttosingleparts",
                {
                    "INPUT": dissolve_layer,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context,
                feedback,
            )

            with edit(copied_layer):
                copied_layer.deleteFeatures(input_layer.selectedFeatureIds())

                fid_field_idx: int = copied_layer.fields().indexOf("fid")
                change_fid: bool = fid_field_idx != -1

                running_fid = 0
                if change_fid:
                    running_fid = copied_layer.maximumValue(fid_field_idx)

                combined_feature: QgsFeature
                for combined_feature in combined_features_layer.getFeatures():
                    if change_fid:
                        combined_feature.setAttribute("fid", running_fid + 1)
                        running_fid += 1
                    copied_layer.addFeature(QgsFeature(combined_feature))

        input_layer.removeSelection()

        check_feature_count_difference(
            input_layer, copied_layer, warn_if_different=False
        )

        feedback.setProgressText("Eliminating features")

        # Delete small lakes
        eliminate_expression = QgsExpression(f"$area >= {min_lake_area}")
        no_small_lakes = copied_layer.materialize(
            QgsFeatureRequest(eliminate_expression)
        )

        # Delete small islands
        eliminated_layer = get_subalgorithm_result_layer(
            "native:deleteholes",
            {
                "INPUT": no_small_lakes,
                "MIN_AREA": min_island_area,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        check_feature_count_difference(
            no_small_lakes, eliminated_layer, warn_if_different=True
        )

        feedback.setProgressText("Extracting islands")

        island_layer = extract_interior_rings_from_layer(eliminated_layer)
        # Extract all islands from the layer
        # island_layer = get_subalgorithm_result_layer(
        #     "yleistys_qgis_plugin:extractinteriorrings",
        #     {
        #         "INPUT": eliminated_layer,
        #         "FEATURE_PER_RING": True,
        #         "OUTPUT": "TEMPORARY_OUTPUT",
        #     },
        #     context,
        #     feedback,
        # )

        feedback.pushInfo(f"Extracted {island_layer.featureCount()} islands.")

        generalized_islands = get_subalgorithm_result_layer(
            "yleistys_qgis_plugin:generalizeislands",
            {
                "INPUT": island_layer,
                "MIN_WIDTH": 185,
                "MIN_ELONGATION": 3.349,
                "MAX_SIMPLIFICATION_TOLERANCE": max_simplification_tolerance,
                "EXAGGERATE_BY": 3,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        check_feature_count_difference(
            island_layer, generalized_islands, warn_if_different=True
        )

        # Delete all islands from the lakes.  This is done because
        # islands and lakes need to be processed differently.
        # The islands will be added back in later.
        exterior_ring_lakes = get_subalgorithm_result_layer(
            "native:deleteholes",
            {
                "INPUT": eliminated_layer,
                "MIN_AREA": 0.0,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        check_feature_count_difference(
            exterior_ring_lakes, eliminated_layer, warn_if_different=True
        )

        if exaggerate_thin_parts:
            feedback.setProgressText("Exaggerating thin parts")

            exterior_ring_lakes = get_subalgorithm_result_layer(
                "yleistys_qgis_plugin:exaggeratethinparts",
                {
                    "INPUT": exterior_ring_lakes,
                    "WIDTH_TOLERANCE": 20,
                    "EXAGGERATE_BY": 3,
                    "MAX_THINNESS": 0.5,
                    "MIN_AREA": 200,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context,
                feedback,
            )

            check_feature_count_difference(
                exterior_ring_lakes, exterior_ring_lakes, warn_if_different=True
            )

        # After exaggerating thin parts simplify and smooth lakes

        feedback.setProgressText("Simplifying and smoothing lakes")

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

        total = (
            100.0 / exterior_ring_lakes.featureCount()
            if exterior_ring_lakes.featureCount()
            else 0
        )

        with edit(exterior_ring_lakes):
            features: QgsFeatureIterator = exterior_ring_lakes.getFeatures()
            for current, feature in enumerate(features):
                if feedback.isCanceled():
                    break

                geom = feature.geometry()

                vertex_count: int = geom.constGet().nCoordinates()
                add_to_tolerance: float = vertex_count / 200
                add_to_tolerance = min(
                    add_to_tolerance, max_simplification_tolerance - 1
                )

                simplifier.setTolerance(1 + add_to_tolerance)

                new_geom = simplifier.simplify(geom)
                new_geom.removeDuplicateNodes()
                smoothed_geom = new_geom.smooth()

                exterior_ring_lakes.changeGeometry(feature.id(), smoothed_geom)

                feedback.setProgress(int(current * total))

        feedback.pushInfo(
            f"Lake features after simplifying and smoothing: {exterior_ring_lakes.featureCount()}"
        )

        # Both lakes and islands have been processed,
        # add islands back in

        feedback.setProgressText("Adding generalized islands")

        difference_layer = get_subalgorithm_result_layer(
            "native:difference",
            {
                "INPUT": exterior_ring_lakes,
                "OVERLAY": generalized_islands,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
            context,
            feedback,
        )

        check_feature_count_difference(
            difference_layer, exterior_ring_lakes, warn_if_different=True
        )

        if QgsWkbTypes.isSingleType(input_layer.wkbType()):
            feedback.setProgressText("Converting to single polygons")
            # the difference algorithm changes the geometries to
            # multipolygons, convert back to single polygon
            generalized_lakes = get_subalgorithm_result_layer(
                "native:multiparttosingleparts",
                {
                    "INPUT": difference_layer,
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context,
                feedback,
            )
            check_feature_count_difference(
                difference_layer, generalized_lakes, warn_if_different=True
            )
        else:
            generalized_lakes = difference_layer

        feedback.setProgress(0)
        total = (
            100.0 / generalized_lakes.featureCount()
            if generalized_lakes.featureCount()
            else 0
        )

        for current, feature in enumerate(generalized_lakes.getFeatures()):
            if feedback.isCanceled():
                break

            sink.addFeature(QgsFeature(feature), QgsFeatureSink.Flag.FastInsert)

            feedback.setProgress(int(current * total))

        return {self.OUTPUT: dest_id}
