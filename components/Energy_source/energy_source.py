"""
Battery component
"""
import openmdao.api as om
from components.Energy_source.Battery.Performances.battery_performance import ComputeBatteryPerfo
from components.Energy_source.Battery.Weight.battery_weight import ComputeBatteryWeight
from components.Energy_source.Battery.Constraints.battery_constraints import BatteryConstraints


class Battery(om.Group):
    """
    Group containing the Battery MDA.
    """

    def setup(self):
        self.add_subsystem("compute_weight", ComputeBatteryWeight(), promotes=["*"])
        self.add_subsystem("compute_perfo", ComputeBatteryPerfo(), promotes=["*"])

        # Constraints
        self.add_subsystem("constraints", BatteryConstraints(), promotes=["*"])

