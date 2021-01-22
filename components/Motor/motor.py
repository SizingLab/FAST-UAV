"""
Motor component
"""
import openmdao.api as om
#from Propeller.Geometry.motor_geometry import ComputeMotorGeometryMR
from Motor.Performances.motor_performance import ComputeMotorPerfoMR
from Motor.Weight.motor_weight import ComputeMotorWeightMR
from Motor.Constraints.motor_constraints import MotorConstraints

class MotorMR(om.Group):
    """
    Group containing the Motor MDA.
    """

    def setup(self):
        #self.add_subsystem("compute_geom", ComputePropellerGeometryMR(), promotes=["*"])
        self.add_subsystem("compute_perfo", ComputeMotorPerfoMR(), promotes=["*"])
        self.add_subsystem("compute_weight", ComputeMotorWeightMR(), promotes=["*"])

        # TO BE ADDED : GEARBOX MODEL with condition (if gearbox_mode == true)
        # self.add_subsystem("gearbox_model", ComputeGearboxParameters(), promotes=["*"])

        # Constraints
        self.add_subsystem("constraints", MotorConstraints(), promotes=["*"])