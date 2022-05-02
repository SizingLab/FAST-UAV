"""
Battery component
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.propulsion.energy.battery.definition_parameters import (
    BatteryDefinitionParameters,
)
from fastuav.models.propulsion.energy.battery.estimation_models import BatteryEstimationModels
from fastuav.models.propulsion.energy.battery.catalogue import BatteryCatalogueSelection
from fastuav.models.propulsion.energy.battery.performance_analysis import BatteryPerformanceGroup
from fastuav.models.propulsion.energy.battery.constraints import BatteryConstraints


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.battery")
class Battery(om.Group):
    """
    Group containing the Battery MDA.
    """

    def initialize(self):
        self.options.declare("off_the_shelf", default=False, types=bool)

    def setup(self):
        self.add_subsystem("definition_parameters", BatteryDefinitionParameters(), promotes=["*"])
        self.add_subsystem("estimation_models", BatteryEstimationModels(), promotes=["*"])
        self.add_subsystem("catalogue_selection" if self.options["off_the_shelf"] else "skip_catalogue_selection",
                           BatteryCatalogueSelection(off_the_shelf=self.options["off_the_shelf"]),
                           promotes=["*"],
        )
        self.add_subsystem("performance_analysis", BatteryPerformanceGroup(), promotes=["*"])
        self.add_subsystem("constraints", BatteryConstraints(), promotes=["*"])
