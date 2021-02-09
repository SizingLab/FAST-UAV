"""
Motor component
"""
import openmdao.api as om
from fastoad.models.options import OpenMdaoOptionDispatcherGroup
from models.Motor.Characteristics.motor_characteristics import ComputeMotorCharacteristics
from models.Motor.Performances.motor_performance import ComputeMotorPerfo
from models.Motor.Weight.motor_weight import ComputeMotorWeight
from models.Motor.Constraints.motor_constraints import MotorConstraints
from models.Motor.Gearbox.gearbox_model import ComputeGearboxCharacteristics
from models.Motor.DecisionTree.motor_catalog import MotorDecisionTree

class Motor(om.Group):
    """
    Group containing the Motor MDA.
    """

    def initialize(self):
        self.options.declare("use_catalogues", default=True, types=bool)
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        self.add_subsystem("characteristics", ComputeMotorCharacteristics(use_gearbox=self.options["use_gearbox"]),
                           promotes=["*"])
        self.add_subsystem("weight", ComputeMotorWeight(), promotes=["*"])

        # Add decision tree regressor for catalogue selection if specified by user ('use_catalogues' = true)
        if self.options["use_catalogues"]:
            self.add_subsystem("catalogue_selection", MotorDecisionTree(), promotes=["*"])

        self.add_subsystem("performances", ComputeMotorPerfo(use_gearbox=self.options["use_gearbox"], use_catalogues=self.options["use_catalogues"]), promotes=["*"])

        # Add gearbox model if specified by user ('use_gearbox' = true)
        if self.options["use_gearbox"]:
            self.add_subsystem("gearbox", ComputeGearboxCharacteristics(), promotes=["*"])

        # Constraints
        self.add_subsystem("define_constraints", MotorConstraints(use_catalogues=self.options['use_catalogues']), promotes=["*"])