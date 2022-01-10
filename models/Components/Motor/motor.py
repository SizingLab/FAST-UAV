"""
Motor component
"""
import fastoad.api as oad
import openmdao.api as om
from models.Components.Motor.definition_parameters import MotorDefinitionParameters
from models.Components.Motor.estimation.models import MotorEstimationModels
from models.Components.Motor.estimation.catalogue import MotorCatalogueSelection
from models.Components.Motor.performances import MotorPerfos
from models.Components.Motor.constraints import MotorConstraints
from models.Components.Motor.Gearbox.models import Gearbox, NoGearbox


@oad.RegisterOpenMDAOSystem("motor")
class Motor(om.Group):
    """
    Group containing the Motor MDA.
    """

    def initialize(self):
        self.options.declare("use_catalogue", default=True, types=bool)
        self.options.declare("use_gearbox", default=True, types=bool)

    def setup(self):
        # Gearbox
        if self.options["use_gearbox"]:
            self.add_subsystem("gearbox", Gearbox(), promotes=["*"])
        else:
            self.add_subsystem("no_gearbox", NoGearbox(), promotes=['*'])

        # Motor
        self.add_subsystem('definition_parameters', MotorDefinitionParameters(), promotes=["*"])
        estimation = self.add_subsystem('estimation', om.Group(), promotes=["*"])
        estimation.add_subsystem("models", MotorEstimationModels(), promotes=["*"])
        estimation.add_subsystem('catalogue' if self.options["use_catalogue"] else 'no_catalogue',
                             MotorCatalogueSelection(use_catalogue=self.options['use_catalogue']), promotes=["*"])
        self.add_subsystem("performances", MotorPerfos(), promotes=["*"])
        self.add_subsystem("constraints", MotorConstraints(), promotes=["*"])