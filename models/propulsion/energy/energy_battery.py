"""
Battery component
"""
import fastoad.api as oad
import openmdao.api as om
from models.propulsion.energy.battery.definition_parameters import BatteryDefinitionParameters
from models.propulsion.energy.battery.estimation.models import BatteryEstimationModels
from models.propulsion.energy.battery.estimation.catalogue import BatteryCatalogueSelection
from models.propulsion.energy.battery.performances import BatteryPerfos
from models.propulsion.energy.battery.constraints import BatteryConstraints


@oad.RegisterOpenMDAOSystem("propulsion.battery")
class Battery(om.Group):
    """
    Group containing the Battery MDA.
    """

    def initialize(self):
        self.options.declare("use_catalogue", default=False, types=bool)

    def setup(self):
        self.add_subsystem(
            "definition_parameters", BatteryDefinitionParameters(), promotes=["*"]
        )
        estimation = self.add_subsystem("estimation", om.Group(), promotes=["*"])
        estimation.add_subsystem("models", BatteryEstimationModels(), promotes=["*"])
        estimation.add_subsystem(
            "catalogue" if self.options["use_catalogue"] else "no_catalogue",
            BatteryCatalogueSelection(use_catalogue=self.options["use_catalogue"]),
            promotes=["*"],
        )
        self.add_subsystem("performances", BatteryPerfos(), promotes=["*"])
        self.add_subsystem("constraints", BatteryConstraints(), promotes=["*"])
