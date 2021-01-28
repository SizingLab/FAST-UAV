"""
Motor component
"""
import openmdao.api as om
from fastoad.models.options import OpenMdaoOptionDispatcherGroup
from components.Motor.Performances.motor_performance import ComputeMotorPerfoMR
from components.Motor.Weight.motor_weight import ComputeMotorWeightMR
from components.Motor.Constraints.motor_constraints import MotorConstraints
from components.Motor.Gearbox.gearbox_model import ComputeGearboxParameters

class MotorMR(OpenMdaoOptionDispatcherGroup):
    """
    Group containing the Motor MDA.
    """

    def initialize(self):
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        self.add_subsystem("compute_perfo", ComputeMotorPerfoMR(), promotes=["*"])
        self.add_subsystem("compute_weight", ComputeMotorWeightMR(), promotes=["*"])

        # Add gearbox model if specified by user ('use_gearbox' = true)
        if self.options["use_gearbox"]:
            self.add_subsystem("gearbox_model", ComputeGearboxParameters(), promotes=["*"])

        # Constraints
        self.add_subsystem("constraints", MotorConstraints(), promotes=["*"])