"""
Battery component
"""
import fastoad.api as oad
import openmdao.api as om
from models.Energy_source.Battery.Scaling.battery_scaling import BatteryScaling
from models.Energy_source.Battery.Performances.battery_performance import BatteryPerfos
from models.Energy_source.Battery.Constraints.battery_constraints import BatteryConstraints
from models.Energy_source.Battery.DecisionTree.battery_catalog import BatteryCatalogueSelection


@oad.RegisterOpenMDAOSystem("energy.battery")
class Battery(om.Group):
    """
    Group containing the Battery MDA.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        self.add_subsystem("scaling", BatteryScaling(), promotes=["*"])
        self.add_subsystem("selection", BatteryCatalogueSelection(use_catalogue=self.options['use_catalogue']),
                           promotes=["*"])
        self.add_subsystem("performances", BatteryPerfos(), promotes=["*"])
        self.add_subsystem("constraints", BatteryConstraints(), promotes=["*"])


