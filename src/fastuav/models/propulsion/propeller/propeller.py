"""
Propeller component
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.propulsion.propeller.definition_parameters import PropellerDefinitionParameters
from fastuav.models.propulsion.propeller.estimation.models import PropellerEstimationModels
from fastuav.models.propulsion.propeller.estimation.catalogue import PropellerCatalogueSelection
from fastuav.models.propulsion.propeller.performances import PropellerPerfos
from fastuav.models.propulsion.propeller.constraints import PropellerConstraints


@oad.RegisterOpenMDAOSystem("propulsion.propeller")
class Propeller(om.Group):
    """
    Group containing the Propeller MDA.
    """

    def initialize(self):
        self.options.declare("use_catalogue", default=False, types=bool)

    def setup(self):
        self.add_subsystem("definition_parameters", PropellerDefinitionParameters(), promotes=["*"])
        estimation = self.add_subsystem("estimation", om.Group(), promotes=["*"])
        estimation.add_subsystem("models", PropellerEstimationModels(), promotes=["*"])
        estimation.add_subsystem(
            "catalogue" if self.options["use_catalogue"] else "no_catalogue",
            PropellerCatalogueSelection(use_catalogue=self.options["use_catalogue"]),
            promotes=["*"],
        )
        self.add_subsystem("performances", PropellerPerfos(), promotes=["*"])
        self.add_subsystem("constraints", PropellerConstraints(), promotes=["*"])
