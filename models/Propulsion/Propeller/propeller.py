"""
Propeller component
"""
import fastoad.api as oad
import openmdao.api as om
from models.Propulsion.Propeller.definition_parameters import PropellerDefinitionParameters
from models.Propulsion.Propeller.estimation.models import PropellerEstimationModels
from models.Propulsion.Propeller.estimation.catalogue import PropellerCatalogueSelection
from models.Propulsion.Propeller.performances import PropellerPerfos
from models.Propulsion.Propeller.constraints import PropellerConstraints


@oad.RegisterOpenMDAOSystem("propulsion.propeller")
class Propeller(om.Group):
    """
    Group containing the Propeller MDA.
    """

    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        self.add_subsystem(
            "definition_parameters", PropellerDefinitionParameters(), promotes=["*"]
        )
        estimation = self.add_subsystem("estimation", om.Group(), promotes=["*"])
        estimation.add_subsystem("models", PropellerEstimationModels(), promotes=["*"])
        estimation.add_subsystem(
            "catalogue" if self.options["use_catalogue"] else "no_catalogue",
            PropellerCatalogueSelection(use_catalogue=self.options["use_catalogue"]),
            promotes=["*"],
        )
        self.add_subsystem("performances", PropellerPerfos(), promotes=["*"])
        self.add_subsystem("constraints", PropellerConstraints(), promotes=["*"])
