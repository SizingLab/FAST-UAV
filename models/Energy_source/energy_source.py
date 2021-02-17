"""
Battery component
"""
import openmdao.api as om
from models.Energy_source.Battery.Characteristics.battery_characteristics import ComputeBatteryCharacteristics
from models.Energy_source.Battery.Weight.battery_weight import ComputeBatteryWeight
from models.Energy_source.Battery.Geometry.battery_geometry import ComputeBatteryGeometry
from models.Energy_source.Battery.Constraints.battery_constraints import BatteryConstraints
from models.Energy_source.Battery.DecisionTree.battery_catalog import BatteryCatalogueSelection


class Battery(om.Group):
    """
    Group containing the Battery MDA.
    """
    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)

    def setup(self):
        self.add_subsystem("weight", ComputeBatteryWeight(), promotes=["*"])
        self.add_subsystem("characteristics", ComputeBatteryCharacteristics(), promotes=["*"])
        self.add_subsystem("geometry", ComputeBatteryGeometry(), promotes=["*"])

        # Choose between estimated parameters and catalogue components
        self.add_subsystem("catalogue", BatteryCatalogueSelection(use_catalogue=self.options['use_catalogue']),
                           promotes=["*"])

        # Constraints
        self.add_subsystem("define_constraints", BatteryConstraints(), promotes=["*"])

