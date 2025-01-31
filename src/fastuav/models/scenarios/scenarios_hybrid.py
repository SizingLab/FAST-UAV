"""
Sizing scenarios definition for hybrid VTOL drones.
The sizing scenarios return the thrusts and loads requirements to size the UAV.
The sizing scenarios are extracted from a sizing mission defined by the user.
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.mtow.mtow import MtowGuess
from fastuav.models.geometry.geometry_fixedwing import ProjectedAreasGuess
from fastuav.models.aerodynamics.aerodynamics_fixedwing import SpanEfficiency, InducedDragConstant
from fastuav.models.scenarios.thrust.takeoff import VerticalTakeoffThrust, LauncherTakeoff
from fastuav.models.scenarios.thrust.cruise import FixedwingCruiseThrust, NoCruise
from fastuav.models.scenarios.thrust.climb import MultirotorClimbThrust, FixedwingClimbThrust
from fastuav.models.scenarios.thrust.hover import HoverThrust, NoHover
from fastuav.models.scenarios.wing_loading.wing_loading import (
    WingLoadingCruise,
    WingLoadingStall,
    WingLoadingSelection,
)


@oad.RegisterOpenMDAOSystem("fastuav.scenarios.hybrid")
class SizingScenariosHybrid(om.Group):
    """
    Sizing scenarios definition for hybrid VTOL configurations
    """

    def setup(self):
        preliminary = self.add_subsystem("preliminary", om.Group(), promotes=["*"])
        preliminary.add_subsystem("mtow_guess", MtowGuess(), promotes=["*"])
        preliminary.add_subsystem("span_efficiency", SpanEfficiency(), promotes=["*"])
        preliminary.add_subsystem("induced_drag_constant", InducedDragConstant(), promotes=["*"])

        wingloading = self.add_subsystem("wing_loading", om.Group(), promotes=["*"])
        wingloading.add_subsystem("stall", WingLoadingStall(), promotes=["*"])
        wingloading.add_subsystem("cruise", WingLoadingCruise(), promotes=["*"])
        wingloading.add_subsystem("selection", WingLoadingSelection(), promotes=["*"])

        self.add_subsystem("areas_guess", ProjectedAreasGuess(), promotes=["*"])

        thrust = self.add_subsystem("thrust", om.Group(), promotes=["*"])
        multirotor = thrust.add_subsystem("multirotor", om.Group(), promotes=["*"])
        multirotor.add_subsystem("takeoff", VerticalTakeoffThrust(), promotes=["*"])
        multirotor.add_subsystem("climb", MultirotorClimbThrust(), promotes=["*"])
        multirotor.add_subsystem("hover", HoverThrust(), promotes=["*"])
        multirotor.add_subsystem("no_cruise", NoCruise(), promotes=["*"])
        fixedwing = thrust.add_subsystem("fixedwing", om.Group(), promotes=["*"])
        fixedwing.add_subsystem("takeoff", LauncherTakeoff(), promotes=["*"])
        fixedwing.add_subsystem("climb", FixedwingClimbThrust(), promotes=["*"])
        fixedwing.add_subsystem("no_hover", NoHover(), promotes=["*"])
        fixedwing.add_subsystem("cruise", FixedwingCruiseThrust(), promotes=["*"])


