"""
Safety module
"""
import fastoad.api as oad
import openmdao.api as om
from models.Add_ons.Safety.Mission.mission import Mission
from models.Add_ons.Safety.FailureMode.failure_mode import FailureMode


@oad.RegisterOpenMDAOSystem("addons.safety")
class Safety(om.Group):
    """
    Group containing the safety requirements components
    """
    def setup(self):
        self.add_subsystem("mission", Mission(), promotes=['*'])
        #self.add_subsystem("failure_mode", FailureMode(), promotes=['*'])








