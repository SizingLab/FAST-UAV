"""
Battery component
"""
import openmdao.api as om
from Battery.Performances.battery_performance import ComputeBatteryPerfo
from Battery.Weight.battery_weight import ComputeBatteryWeight
from Battery.Constraints.battery_constraints import BatteryConstraints

class Battery(om.Group):
    """
    Group containing the Battery MDA.
    """

    def setup(self):
        self.add_subsystem("compute_weight", ComputeBatteryWeight(), promotes=["*"])
        self.add_subsystem("compute_perfo", ComputeBatteryPerfo(), promotes=["*"])

        # Constraints
        self.add_subsystem("constraints", BatteryConstraints(), promotes=["*"])