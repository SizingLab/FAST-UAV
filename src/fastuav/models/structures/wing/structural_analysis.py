"""
Structural analysis for the wing
"""

import openmdao.api as om
import numpy as np
from fastuav.utils.constants import MR_PROPULSION


class WingStructuralAnalysisModels:
    """
    Structural analysis models for the wing.
    """

    @staticmethod
    def i_beam_stress(M_root, h_web, k_spar, k_flange=0.1):
        """
        Structural analysis for a cantilever I-shaped beam with a bending moment at root M_root.
        The model assumes that the bending moment is entirely reacted by the spar flanges.

        :param M_root: Bending moment at root [N.m]
        :param h_web: Depth of the web [m]
        :param k_spar: Depth ratio of the spar (k_a = flange depth / web depth) [-]
        :param k_flange: Aspect ratio of the flange (b_flange = a_flange / k_flange) [-]
        :return sig_root: Stress at root [Pa]
        """
        sig_root = M_root * (1 + k_spar) / (h_web**3 * k_spar**2 * (1 + k_spar**2 / 3) / k_flange)  # [Pa]
        return sig_root

    @staticmethod
    def pipe_stress(M_root, d_out, k_spar):
        """
        Structural analysis for a cantilever pipe with a bending moment at root M_root.

        :param M_root: Bending moment at root [N.m]
        :param d_out: External diameter of the spar [m]
        :param k_spar: Diameter ratio of the spar (k_d = d_in / d_out) [-]
        :return sig_root: Stress at root [Pa]
        """
        sig_root = (32 * M_root) / (np.pi * (1 - k_spar**4) * d_out**3)
        return sig_root


class SparsStressVTOL(om.ExplicitComponent):
    """
    Stress calculation during vertical takeoff with VTOL propellers.
    """

    def initialize(self):
        self.options.declare("spar_model", default="pipe", values=["pipe", "I_beam"])
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        spar_model = self.options["spar_model"]
        propulsion_id = self.options["propulsion_id"]

        self.add_input("data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id, val=np.nan, units="N")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:geometry:%s:propeller:y" % propulsion_id, val=np.nan, units="m")
        if spar_model == "pipe":
            self.add_input("data:structures:wing:spar:diameter:outer", val=np.nan, units="m")
            self.add_input("data:structures:wing:spar:diameter:k", val=0.9, units=None)
        else:
            self.add_input("data:structures:wing:spar:web:depth", val=np.nan, units="m")
            self.add_input("data:structures:wing:spar:depth:k", val=0.1, units=None)
        self.add_output("data:structures:wing:spar:stress:VTOL", units="N/m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        spar_model = self.options["spar_model"]
        propulsion_id = self.options["propulsion_id"]
        F_pro_to = inputs["data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id]
        N_pro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        y_pro_MR = inputs["data:geometry:%s:propeller:y" % propulsion_id]

        # Half-wing supports half of the total thrust at takeoff
        M_root = F_pro_to * N_pro / 2 * y_pro_MR  # bending moment at spar's root [N.m]

        if spar_model == "pipe":
            d_out = inputs["data:structures:wing:spar:diameter:outer"]  # outer diameter [m]
            k_spar = inputs["data:structures:wing:spar:diameter:k"]  # aspect ratio of the spar [-]
            sig_root = WingStructuralAnalysisModels.pipe_stress(M_root, d_out, k_spar)
        else:
            h_web = inputs["data:structures:wing:spar:web:depth"]  # distance between the two flanges [m]
            k_spar = inputs["data:structures:wing:spar:depth:k"]  # aspect ratio of the spar [-]
            sig_root = WingStructuralAnalysisModels.i_beam_stress(M_root, h_web, k_spar)

        outputs["data:structures:wing:spar:stress:VTOL"] = sig_root