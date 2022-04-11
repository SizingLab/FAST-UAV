"""
Sizing scenarios definition
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.scenarios.atmosphere import Atmosphere
from fastuav.models.scenarios.multirotor.preliminary import MTOWguess, NumberPropellersMR, BodySurfacesMR
from fastuav.models.scenarios.multirotor.thrust import ThrustClimbMR, ThrustCruiseMR, ThrustHoverMR, ThrustTakeOffMR


@oad.RegisterOpenMDAOSystem("scenarios.multirotor")
class SizingScenariosMultirotor(om.Group):
    """
    Sizing scenarios definition for multirotor configurations
    """

    def setup(self):
        preliminary = self.add_subsystem("preliminary", om.Group(), promotes=["*"])
        preliminary.add_subsystem("MTOW_guess", MTOWguess(), promotes=["*"])
        preliminary.add_subsystem("number_props", NumberPropellersMR(), promotes=["*"])
        preliminary.add_subsystem("body_surface", BodySurfacesMR(), promotes=["*"])

        self.add_subsystem("atmosphere", Atmosphere(), promotes=["*"])

        thrust = self.add_subsystem("thrust", om.Group(), promotes=["*"])
        thrust.add_subsystem("hover", ThrustHoverMR(), promotes=["*"])
        thrust.add_subsystem("takeoff", ThrustTakeOffMR(), promotes=["*"])
        thrust.add_subsystem("climb", ThrustClimbMR(), promotes=["*"])
        thrust.add_subsystem("cruise", ThrustCruiseMR(), promotes=["*"])
