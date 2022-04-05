"""
Sizing scenarios definition
"""
import fastoad.api as oad
import openmdao.api as om
from models.Scenarios.atmosphere import Atmosphere
from models.Scenarios.architecture import MTOWguess, SpanEfficiency, InducedDragConstant, NumberPropellersMR, BodySurfacesMR
from models.Scenarios.multirotor.thrust import ThrustClimbMR, ThrustCruiseMR, ThrustHoverMR, ThrustTakeOffMR
from models.Scenarios.fixedwing.thrust import ThrustClimbFW, ThrustCruiseFW, ThrustTakeOffFW, ThrustHoverFW
from models.Scenarios.fixedwing.wing_loading import WingLoadingCruise, WingLoadingStall, WingLoadingSelection


@oad.RegisterOpenMDAOSystem("sizing_scenarios.multirotor")
class SizingScenariosMultirotor(om.Group):
    """
    Sizing scenarios definition for multirotor configurations
    """

    def setup(self):
        self.add_subsystem("atmosphere", Atmosphere(), promotes=["*"])

        architecture = self.add_subsystem("architecture", om.Group(), promotes=["*"])
        architecture.add_subsystem("MTOW_guess", MTOWguess(), promotes=["*"])
        architecture.add_subsystem("number_props", NumberPropellersMR(), promotes=["*"])
        architecture.add_subsystem("body_surface", BodySurfacesMR(), promotes=["*"])

        thrust = self.add_subsystem("thrust", om.Group(), promotes=["*"])
        thrust.add_subsystem("hover", ThrustHoverMR(), promotes=["*"])
        thrust.add_subsystem("takeoff", ThrustTakeOffMR(), promotes=["*"])
        thrust.add_subsystem("climb", ThrustClimbMR(), promotes=["*"])
        thrust.add_subsystem("cruise", ThrustCruiseMR(), promotes=["*"])


@oad.RegisterOpenMDAOSystem("sizing_scenarios.fixedwing")
class SizingScenariosFixedWing(om.Group):
    """
    Sizing scenarios definition for fixed wing configurations
    """

    def setup(self):
        self.add_subsystem("atmosphere", Atmosphere(), promotes=["*"])

        architecture = self.add_subsystem("architecture", om.Group(), promotes=["*"])
        architecture.add_subsystem("MTOW_guess", MTOWguess(), promotes=["*"])
        architecture.add_subsystem("span_efficiency", SpanEfficiency(), promotes=["*"])
        architecture.add_subsystem("induced_drag_constant", InducedDragConstant(), promotes=["*"])

        wingloading = self.add_subsystem("wing_loading", om.Group(), promotes=["*"])
        wingloading.add_subsystem("stall", WingLoadingStall(), promotes=["*"])
        wingloading.add_subsystem("cruise", WingLoadingCruise(), promotes=["*"])
        wingloading.add_subsystem("selection", WingLoadingSelection(), promotes=["*"])

        thrust = self.add_subsystem("thrust", om.Group(), promotes=["*"])
        thrust.add_subsystem("climb", ThrustClimbFW(), promotes=["*"])
        thrust.add_subsystem("cruise", ThrustCruiseFW(), promotes=["*"])
        thrust.add_subsystem("takeoff", ThrustTakeOffFW(), promotes=["*"])
        thrust.add_subsystem("hover", ThrustHoverFW(), promotes=["*"])


