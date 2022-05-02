"""
Motor component
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.propulsion.motor.definition_parameters import MotorDefinitionParameters
from fastuav.models.propulsion.motor.estimation_models import MotorEstimationModels
from fastuav.models.propulsion.motor.catalogue import MotorCatalogueSelection
from fastuav.models.propulsion.motor.performance_analysis import MotorPerformanceGroup
from fastuav.models.propulsion.motor.constraints import MotorConstraints


@oad.RegisterOpenMDAOSystem("fastuav.propulsion.motor")
class Motor(om.Group):
    """
    Group containing the Motor MDA.
    """

    def initialize(self):
        self.options.declare("off_the_shelf", default=False, types=bool)

    def setup(self):
        self.add_subsystem("definition_parameters", MotorDefinitionParameters(), promotes=["*"])
        self.add_subsystem("estimation_models", MotorEstimationModels(), promotes=["*"])
        self.add_subsystem("catalogue_selection" if self.options["off_the_shelf"] else "skip_catalogue_selection",
                           MotorCatalogueSelection(off_the_shelf=self.options["off_the_shelf"]),
                           promotes=["*"],
        )
        self.add_subsystem("performance_analysis", MotorPerformanceGroup(), promotes=["*"])
        self.add_subsystem("constraints", MotorConstraints(), promotes=["*"])
