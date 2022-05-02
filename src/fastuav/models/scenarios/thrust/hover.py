"""
Hover scenarios
"""

import numpy as np
from scipy.constants import g
import openmdao.api as om
from fastuav.utils.constants import MR_PROPULSION, FW_PROPULSION


class HoverThrust(om.ExplicitComponent):
    """
    Thrust to maintain hover.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:weights:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
        self.add_output("data:propulsion:%s:propeller:thrust:hover" % propulsion_id, units="N")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        Mtotal_guess = inputs["data:weights:mtow:guess"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]

        F_pro_hov = Mtotal_guess * g / Npro  # [N] Thrust per propeller

        outputs["data:propulsion:%s:propeller:thrust:hover" % propulsion_id] = F_pro_hov


class NoHover(om.ExplicitComponent):
    """
    Simple component to declare the absence of hover scenario.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_output("data:propulsion:%s:propeller:thrust:hover" % propulsion_id, units="N")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        outputs["data:propulsion:%s:propeller:thrust:hover" % propulsion_id] = 0.0