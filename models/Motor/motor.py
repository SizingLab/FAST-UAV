"""
Motor component
"""
import fastoad.api as oad
import openmdao.api as om
from models.Motor.Scaling.motor_scaling import MotorScaling
from models.Motor.DecisionTree.motor_catalog import MotorCatalogueSelection
from models.Motor.Performances.motor_performance import MotorPerfos
from models.Motor.Constraints.motor_constraints import MotorConstraints
from models.Motor.Gearbox.gearbox_model import Gearbox, NoGearbox


@oad.RegisterOpenMDAOSystem("motor")
class Motor(om.Group):
    """
    Group containing the Motor MDA.
    """

    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        self.add_subsystem("scaling", MotorScaling(use_gearbox=self.options["use_gearbox"]),promotes=["*"])
        self.add_subsystem("catalogue", MotorCatalogueSelection(use_catalogue=self.options['use_catalogue']), promotes=["*"])
        self.add_subsystem("performances", MotorPerfos(use_gearbox=self.options["use_gearbox"]), promotes=["*"])
        if self.options["use_gearbox"]:
            self.add_subsystem("gearbox", Gearbox(), promotes=["*"])
        else:
            self.add_subsystem("no_gearbox", NoGearbox(), promotes=['*'])
        self.add_subsystem("constraints", MotorConstraints(), promotes=["*"])