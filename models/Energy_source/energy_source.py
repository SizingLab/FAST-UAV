"""
Battery component
"""
import openmdao.api as om
from models.Energy_source.Battery.Characteristics.battery_characteristics import ComputeBatteryCharacteristics
from models.Energy_source.Battery.Weight.battery_weight import ComputeBatteryWeight
from models.Energy_source.Battery.Constraints.battery_constraints import BatteryConstraints
from models.Energy_source.Battery.DecisionTree.battery_catalog import BatteryDecisionTree


class Battery(om.Group):
    """
    Group containing the Battery MDA.
    """
    def initialize(self):
        self.options.declare("use_catalogues", default=True, types=bool)

    def setup(self):
        self.add_subsystem("weight", ComputeBatteryWeight(), promotes=["*"])
        self.add_subsystem("characteristics", ComputeBatteryCharacteristics(), promotes=["*"])

        # Add decision tree regressor for catalogue selection if specified by user ('use_catalogues' = true)
        if self.options["use_catalogues"]:
            self.add_subsystem("catalogue_selection", BatteryDecisionTree(), promotes=["*"])

        # Constraints
        self.add_subsystem("define_constraints", BatteryConstraints(use_catalogues=self.options['use_catalogues']), promotes=["*"])

