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
        self.add_input("data:system:MTOW:k", val=np.nan, units=None)
        self.add_input("specifications:payload:mass", val=np.nan, units="kg")
        self.add_output("data:system:MTOW:guess", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        k_M = inputs["data:system:MTOW:k"]
        M_load = inputs["specifications:payload:mass"]

        Mtotal_guess = k_M * M_load  # [kg] Estimate of the total mass (or equivalent weight of dynamic scenario)

        outputs["data:system:MTOW:guess"] = Mtotal_guess

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k_M = inputs["data:system:MTOW:k"]
        M_load = inputs["specifications:payload:mass"]
        partials["data:system:MTOW:guess", "data:system:MTOW:k"] = M_load
        partials["data:system:MTOW:guess", "specifications:payload:mass"] = k_M


class SpanEfficiency(om.ExplicitComponent):
    """
    Computes the span efficiency
    """

    def setup(self):
        self.add_input("data:airframe:wing:AR", val=np.nan, units=None)
        self.add_output("data:airframe:aerodynamics:CDi:e", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_w = inputs["data:airframe:wing:AR"]

        e = 1.78 * (
                    1 - 0.045 * AR_w ** 0.68) \
            - 0.64  # span efficiency factor (empirical estimation for straight wings, Raymer)

        outputs["data:airframe:aerodynamics:CDi:e"] = e

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        AR_w = inputs["data:airframe:wing:AR"]

        partials["data:airframe:aerodynamics:CDi:e",
                 "data:airframe:wing:AR"] = - 1.78 * 0.045 * 0.68 * AR_w ** (0.68 - 1)


class InducedDragConstant(om.ExplicitComponent):
    """
    Computes the induced drag constant
    """

    def setup(self):
        self.add_input("data:airframe:wing:AR", val=np.nan, units=None)
        self.add_input("data:airframe:aerodynamics:CDi:e", val=np.nan, units=None)
        self.add_output("data:airframe:aerodynamics:CDi:K", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_w = inputs["data:airframe:wing:AR"]
        e = inputs["data:airframe:aerodynamics:CDi:e"]

        K = 1 / (np.pi * e * AR_w)  # induced drag constant (correction term for non-elliptical lift distribution)

        outputs["data:airframe:aerodynamics:CDi:K"] = K

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        AR_w = inputs["data:airframe:wing:AR"]
        e = inputs["data:airframe:aerodynamics:CDi:e"]

        partials["data:airframe:aerodynamics:CDi:K",
                 "data:airframe:wing:AR"] = - 1 / (np.pi * e * AR_w ** 2)
        partials["data:airframe:aerodynamics:CDi:K",
                 "data:airframe:aerodynamics:CDi:e"] = - 1 / (np.pi * e ** 2 * AR_w)


class NumberPropellersMR(om.ExplicitComponent):
    """
    Computes the total number of propellers for multirotors
    """

    def setup(self):
        self.add_input("data:airframe:arms:prop_per_arm", val=np.nan, units=None)
        self.add_input("data:airframe:arms:number", val=np.nan, units=None)
        self.add_output("data:propeller:number", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Npro_arm = inputs["data:airframe:arms:prop_per_arm"]
        Narm = inputs["data:airframe:arms:number"]

        Npro = Npro_arm * Narm  # [-] Number of propellers

        outputs["data:propeller:number"] = Npro


class BodySurfacesMR(om.ExplicitComponent):
    """
    Scaling laws for body areas calculation, for multirotor configurations.
    """

    def setup(self):
        self.add_input("data:system:MTOW:guess", val=np.nan, units="kg")
        self.add_input("data:airframe:body:reference:surface:top", val=np.nan, units="m**2")
        self.add_input("data:airframe:body:reference:surface:front", val=np.nan, units="m**2")
        self.add_input("data:system:reference:MTOW", val=np.nan, units="kg")
        self.add_output("data:airframe:body:surface:top", units="m**2")
        self.add_output("data:airframe:body:surface:front", units="m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs["data:system:MTOW:guess"]
        S_top_ref = inputs["data:airframe:body:reference:surface:top"]
        S_front_ref = inputs["data:airframe:body:reference:surface:front"]
        MTOW_ref = inputs["data:system:reference:MTOW"]

        S_top_estimated = S_top_ref * (Mtotal_guess / MTOW_ref) ** (2 / 3)  # [m2] top surface estimation
        S_front_estimated = S_front_ref * (Mtotal_guess / MTOW_ref) ** (2 / 3)  # [m2] front surface estimation

        outputs["data:airframe:body:surface:top"] = S_top_estimated
        outputs["data:airframe:body:surface:front"] = S_front_estimated