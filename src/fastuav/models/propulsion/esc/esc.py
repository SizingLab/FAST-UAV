"""
ESC component
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.propulsion.esc.definition_parameters import ESCDefinitionParameters
from fastuav.models.propulsion.esc.estimation_models import ESCEstimationModels
from fastuav.models.propulsion.esc.catalogue import ESCCatalogueSelection
from fastuav.models.propulsion.esc.performance_analysis import ESCPerformanceGroup
from fastuav.models.propulsion.esc.constraints import ESCConstraints


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.esc")
class ESC(om.Group):
    """
    Group containing the Electronic Speed Controller analysis.
    """

    def initialize(self):
        self.options.declare("off_the_shelf", default=False, types=bool)

    def setup(self):
        self.add_subsystem("definition_parameters", ESCDefinitionParameters(), promotes=["*"])
        self.add_subsystem("estimation_models", ESCEstimationModels(), promotes=["*"])
        self.add_subsystem("catalogue_selection" if self.options["off_the_shelf"] else "skip_catalogue_selection",
                           ESCCatalogueSelection(off_the_shelf=self.options["off_the_shelf"]),
                           promotes=["*"],
        )
        self.add_subsystem("performance_analysis", ESCPerformanceGroup(), promotes=["*"])
        self.add_subsystem("constraints", ESCConstraints(), promotes=["*"])
