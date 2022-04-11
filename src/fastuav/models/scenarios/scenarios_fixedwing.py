"""
Sizing scenarios definition
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.scenarios.atmosphere import Atmosphere
from fastuav.models.scenarios.fixedwing.preliminary import MTOWguess, SpanEfficiency, InducedDragConstant
from fastuav.models.scenarios.fixedwing.thrust import ThrustClimbFW, ThrustCruiseFW, ThrustTakeOffFW
from fastuav.models.scenarios.fixedwing.wing_loading import WingLoadingCruise, WingLoadingStall, WingLoadingSelection


@oad.RegisterOpenMDAOSystem("scenarios.fixedwing")
class SizingScenariosFixedWing(om.Group):
    """
    Sizing scenarios definition for fixed wing configurations
    """

    def setup(self):
        preliminary = self.add_subsystem("preliminary", om.Group(), promotes=["*"])
        preliminary.add_subsystem("MTOW_guess", MTOWguess(), promotes=["*"])
        preliminary.add_subsystem("span_efficiency", SpanEfficiency(), promotes=["*"])
        preliminary.add_subsystem("induced_drag_constant", InducedDragConstant(), promotes=["*"])

        self.add_subsystem("atmosphere", Atmosphere(), promotes=["*"])

        wingloading = self.add_subsystem("wing_loading", om.Group(), promotes=["*"])
        wingloading.add_subsystem("stall", WingLoadingStall(), promotes=["*"])
        wingloading.add_subsystem("cruise", WingLoadingCruise(), promotes=["*"])
        wingloading.add_subsystem("selection", WingLoadingSelection(), promotes=["*"])

        thrust = self.add_subsystem("thrust", om.Group(), promotes=["*"])
        thrust.add_subsystem("climb", ThrustClimbFW(), promotes=["*"])
        thrust.add_subsystem("cruise", ThrustCruiseFW(), promotes=["*"])
        thrust.add_subsystem("takeoff", ThrustTakeOffFW(), promotes=["*"])


