"""
Sizing scenarios definition.
The sizing scenarios return the thrusts and loads requirements to size the UAV.
The sizing scenarios are extracted from a sizing mission defined by the user.
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.mtow.mtow import MTOW_guess
from fastuav.models.geometry.geometry_multirotor import BodyAreas
from fastuav.models.scenarios.thrust.takeoff import VerticalTakeoffThrust
from fastuav.models.scenarios.thrust.cruise import MultirotorCruiseThrust
from fastuav.models.scenarios.thrust.climb import VerticalClimbThrust
from fastuav.models.scenarios.thrust.hover import HoverThrust


@oad.RegisterOpenMDAOSystem("fastuav.scenarios.multirotor")
class SizingScenariosMultirotor(om.Group):
    """
    Sizing scenarios definition for multirotor configurations
    """

    def setup(self):
        preliminary = self.add_subsystem("preliminary", om.Group(), promotes=["*"])
        preliminary.add_subsystem("mtow_guess", MTOW_guess(), promotes=["*"])
        preliminary.add_subsystem("body_areas", BodyAreas(), promotes=["*"])

        thrust = self.add_subsystem("thrust", om.Group(), promotes=["*"])
        thrust.add_subsystem("hover", HoverThrust(), promotes=["*"])
        thrust.add_subsystem("takeoff", VerticalTakeoffThrust(), promotes=["*"])
        thrust.add_subsystem("climb", VerticalClimbThrust(), promotes=["*"])
        thrust.add_subsystem("cruise", MultirotorCruiseThrust(), promotes=["*"])
