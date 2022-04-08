"""
Preliminary calculations.
"""

import numpy as np
import openmdao.api as om


class MTOWguess(om.ExplicitComponent):
    """
    Computes an initial guess for the MTOW
    """

    def setup(self):
        self.add_input("data:weights:MTOW:k", val=np.nan, units=None)
        self.add_input("specifications:payload:mass", val=np.nan, units="kg")
        self.add_output("data:weights:MTOW:guess", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_M = inputs["data:weights:MTOW:k"]
        M_load = inputs["specifications:payload:mass"]

        Mtotal_guess = k_M * M_load  # [kg] Estimate of the total mass (or equivalent weight of dynamic scenario)

        outputs["data:weights:MTOW:guess"] = Mtotal_guess

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_M = inputs["data:weights:MTOW:k"]
        M_load = inputs["specifications:payload:mass"]
        partials["data:weights:MTOW:guess", "data:weights:MTOW:k"] = M_load
        partials["data:weights:MTOW:guess", "specifications:payload:mass"] = k_M


class NumberPropellersMR(om.ExplicitComponent):
    """
    Computes the total number of multirotor propellers
    """

    def setup(self):
        self.add_input("data:geometry:arms:prop_per_arm", val=np.nan, units=None)
        self.add_input("data:geometry:arms:number", val=np.nan, units=None)
        self.add_output("data:propulsion:propeller:number", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Npro_arm = inputs["data:geometry:arms:prop_per_arm"]
        Narm = inputs["data:geometry:arms:number"]

        Npro = Npro_arm * Narm  # [-] Number of propellers

        outputs["data:propulsion:propeller:number"] = Npro


class BodySurfacesMR(om.ExplicitComponent):
    """
    Scaling laws for body areas calculation
    """

    def setup(self):
        self.add_input("data:weights:MTOW:guess", val=np.nan, units="kg")
        self.add_input("data:geometry:body:surface:top:reference", val=np.nan, units="m**2")
        self.add_input("data:geometry:body:surface:front:reference", val=np.nan, units="m**2")
        self.add_input("data:weights:MTOW:reference", val=np.nan, units="kg")
        self.add_output("data:geometry:body:surface:top", units="m**2")
        self.add_output("data:geometry:body:surface:front", units="m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs["data:weights:MTOW:guess"]
        S_top_ref = inputs["data:geometry:body:surface:top:reference"]
        S_front_ref = inputs["data:geometry:body:surface:front:reference"]
        MTOW_ref = inputs["data:weights:MTOW:reference"]

        S_top_estimated = S_top_ref * (Mtotal_guess / MTOW_ref) ** (2 / 3)  # [m2] top surface estimation
        S_front_estimated = S_front_ref * (Mtotal_guess / MTOW_ref) ** (2 / 3)  # [m2] front surface estimation

        outputs["data:geometry:body:surface:top"] = S_top_estimated
        outputs["data:geometry:body:surface:front"] = S_front_estimated