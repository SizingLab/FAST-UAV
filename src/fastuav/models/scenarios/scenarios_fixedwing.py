"""
Sizing scenarios definition.
The sizing scenarios return the thrusts and loads requirements to size the UAV.
The sizing scenarios are extracted from a sizing mission defined by the user.
"""
import fastoad.api as oad
import openmdao.api as om
from fastuav.models.mtow.mtow import MTOW_guess
from fastuav.models.aerodynamics.aerodynamics_fixedwing import SpanEfficiency, InducedDragConstant
from fastuav.models.scenarios.thrust.takeoff import LauncherTakeoff
from fastuav.models.scenarios.thrust.cruise import FixedwingCruiseThrust
from fastuav.models.scenarios.thrust.climb import FixedwingClimbThrust
from fastuav.models.scenarios.thrust.hover import NoHover
from fastuav.models.scenarios.wing_loading.wing_loading import (
    WingLoadingCruise,
    WingLoadingStall,
    WingLoadingSelection,
)


@oad.RegisterOpenMDAOSystem("fastuav.scenarios.fixedwing")
class SizingScenariosFixedWing(om.Group):
    """
    Sizing scenarios definition for fixed wing configurations
    """

    def setup(self):
        preliminary = self.add_subsystem("preliminary", om.Group(), promotes=["*"])
        preliminary.add_subsystem("mtow_guess", MTOW_guess(), promotes=["*"])
        preliminary.add_subsystem("span_efficiency", SpanEfficiency(), promotes=["*"])
        preliminary.add_subsystem("induced_drag_constant", InducedDragConstant(), promotes=["*"])

        wingloading = self.add_subsystem("wing_loading", om.Group(), promotes=["*"])
        wingloading.add_subsystem("stall", WingLoadingStall(), promotes=["*"])
        wingloading.add_subsystem("cruise", WingLoadingCruise(), promotes=["*"])
        wingloading.add_subsystem("selection", WingLoadingSelection(), promotes=["*"])

        thrust = self.add_subsystem("thrust", om.Group(), promotes=["*"])
        thrust.add_subsystem("takeoff", LauncherTakeoff(), promotes=["*"])
        thrust.add_subsystem("climb", FixedwingClimbThrust(), promotes=["*"])
        thrust.add_subsystem("cruise", FixedwingCruiseThrust(), promotes=["*"])
        thrust.add_subsystem("no_hover", NoHover(), promotes=["*"])

