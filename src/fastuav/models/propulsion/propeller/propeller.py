"""
Propeller component
"""
import openmdao.api as om
from fastuav.models.propulsion.propeller.definition_parameters import PropellerDefinitionParameters
from fastuav.models.propulsion.propeller.estimation_models import PropellerEstimationModels
from fastuav.models.propulsion.propeller.catalogue import PropellerCatalogueSelection
from fastuav.models.propulsion.propeller.performance_analysis import PropellerPerformanceGroup
from fastuav.models.propulsion.propeller.constraints import PropellerConstraints


class Propeller(om.Group):
    """
    Group containing the Propeller MDA.
    """

    def initialize(self):
        self.options.declare("off_the_shelf", default=False, types=bool)

    def setup(self):
        self.add_subsystem("definition_parameters", PropellerDefinitionParameters(), promotes=["*"])
        self.add_subsystem("estimation_models",
                           PropellerEstimationModels(),
                           promotes=["*"])
        self.add_subsystem("catalogue_selection" if self.options["off_the_shelf"] else "skip_catalogue_selection",
                           PropellerCatalogueSelection(off_the_shelf=self.options["off_the_shelf"]),
                           promotes=["*"])
        self.add_subsystem("performance_analysis",
                           PropellerPerformanceGroup(),
                           promotes=["*"])
        self.add_subsystem("constraints",
                           PropellerConstraints(),
                           promotes=["*"])
