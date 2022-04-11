"""
Motor component
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.propulsion.motor.definition_parameters import MotorDefinitionParameters
from fastuav.models.propulsion.motor.estimation.models import MotorEstimationModels
from fastuav.models.propulsion.motor.estimation.catalogue import MotorCatalogueSelection
from fastuav.models.propulsion.motor.performances import MotorPerfos
from fastuav.models.propulsion.motor.constraints import MotorConstraints
from fastuav.models.propulsion.motor.gearbox.models import Gearbox, NoGearbox


@oad.RegisterOpenMDAOSystem("propulsion.motor")
class Motor(om.Group):
    """
    Group containing the Motor MDA.
    """

    def initialize(self):
        self.options.declare("use_catalogue", default=False, types=bool)
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        # Motor
        self.add_subsystem("definition_parameters", MotorDefinitionParameters(), promotes=["*"])
        estimation = self.add_subsystem("estimation", om.Group(), promotes=["*"])
        estimation.add_subsystem("models", MotorEstimationModels(), promotes=["*"])
        estimation.add_subsystem(
            "catalogue" if self.options["use_catalogue"] else "no_catalogue",
            MotorCatalogueSelection(use_catalogue=self.options["use_catalogue"]),
            promotes=["*"],
        )
        self.add_subsystem("performances", MotorPerfos(), promotes=["*"])
        self.add_subsystem("constraints", MotorConstraints(), promotes=["*"])

        # Gearbox
        if self.options["use_gearbox"]:
            self.add_subsystem("gearbox", Gearbox(), promotes=["*"])
        else:
            self.add_subsystem("no_gearbox", NoGearbox(), promotes=["*"])
