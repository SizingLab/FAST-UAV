"""
Hybrid VTOL Aerodynamics (external)
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import (
    add_subsystem_with_deviation,
)
from fastuav.models.aerodynamics.aerodynamics_fixedwing import WingParasiticDrag, TailParasiticDrag, FuselageParasiticDrag, ParasiticDragConstraint, MaxLiftToDrag
from fastuav.utils.constants import MR_PROPULSION


class StoppedPropellersAerodynamicsModel:
    """
    Aerodynamics model for the stopped propellers drag
    """

    @staticmethod
    def stopped_propeller_drag_coefficient(alpha, N_blades=3, Cl_alpha=0.04 * 180 / np.pi, A_blade=7):
        """
        Computes the drag coefficient of a stopped VTOL propeller when the aircraft is in forward flight,
        with an angle of attack alpha (rad).
        """
        BAR = N_blades / (np.pi * A_blade)  # [-] blade area ratio
        CD_blade = (0.1 + Cl_alpha ** 2 * alpha ** 2)  # feathered blade drag coefficient
        CD_prop = BAR * CD_blade  # [-] drag coefficient of the propeller
        return CD_prop


@oad.RegisterOpenMDAOSystem("fastuav.aerodynamics.hybrid")
class Aerodynamics(om.Group):
    """
    Group containing the external aerodynamics calculation
    """

    def setup(self):

        # Parasitic drag calculations
        parasitic_drag = self.add_subsystem("parasitic_drag", om.Group(), promotes=["*"])
        parasitic_drag.add_subsystem("wing", WingParasiticDrag(), promotes=["*"])
        parasitic_drag.add_subsystem("horizontal_tail", TailParasiticDrag(tail="horizontal"), promotes=["*"])
        parasitic_drag.add_subsystem("vertical_tail", TailParasiticDrag(tail="vertical"), promotes=["*"])
        parasitic_drag.add_subsystem("fuselage", FuselageParasiticDrag(), promotes=["*"])
        parasitic_drag.add_subsystem("stopped_propellers", StoppedPropellersParasiticDrag(), promotes=["*"])
        add_subsystem_with_deviation(
            parasitic_drag,
            "parasitic_drag",
            ParasiticDrag(),
            uncertain_outputs={"data:aerodynamics:CD0": None},
        )
        parasitic_drag.add_subsystem("constraint", ParasiticDragConstraint(), promotes=["*"])

        # Lift to drag
        self.add_subsystem("lift_to_drag", MaxLiftToDrag(), promotes=["*"])


class ParasiticDrag(om.ExplicitComponent):
    """
    Sums up the individual parasitic drags at cruise conditions
    """

    def setup(self):
        self.add_input("data:aerodynamics:CD0:stopped_propellers", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CD0:wing", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CD0:tail:horizontal", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CD0:tail:vertical", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CD0:fuselage", val=np.nan, units=None)
        self.add_output("data:aerodynamics:CD0", units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        outputs["data:aerodynamics:CD0"] = inputs["data:aerodynamics:CD0:wing"] \
                                           + inputs["data:aerodynamics:CD0:tail:horizontal"] \
                                           + inputs["data:aerodynamics:CD0:tail:vertical"] \
                                           + inputs["data:aerodynamics:CD0:fuselage"]


class StoppedPropellersParasiticDrag(om.ExplicitComponent):
    """
    Computes parasitic drag of VTOL stopped propellers at cruise conditions
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:propulsion:%s:propeller:diameter" % propulsion_id, val=np.nan, units="m")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_output("data:aerodynamics:CD0:stopped_propellers", units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        N_pro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        D_pro = inputs["data:propulsion:%s:propeller:diameter" % propulsion_id]
        S_ref = inputs["data:geometry:wing:surface"]
        alpha = 0.0  # TODO: get trimmed aircraft angle of attack

        # Total area of propellers
        S_pro = N_pro * np.pi * (D_pro / 2) ** 2

        # Parasitic drag coefficient
        CD_0_pro = StoppedPropellersAerodynamicsModel.stopped_propeller_drag_coefficient(alpha) * (S_pro / S_ref)

        outputs["data:aerodynamics:CD0:stopped_propellers"] = CD_0_pro