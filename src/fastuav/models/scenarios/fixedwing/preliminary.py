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


class SpanEfficiency(om.ExplicitComponent):
    """
    Computes the span efficiency
    """

    def setup(self):
        self.add_input("data:geometry:wing:AR", val=np.nan, units=None)
        self.add_output("data:aerodynamics:CDi:e", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_w = inputs["data:geometry:wing:AR"]

        e = 1.78 * (
                    1 - 0.045 * AR_w ** 0.68) \
            - 0.64  # span efficiency factor (empirical estimation for straight wings, Raymer)

        outputs["data:aerodynamics:CDi:e"] = e

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        AR_w = inputs["data:geometry:wing:AR"]

        partials["data:aerodynamics:CDi:e",
                 "data:geometry:wing:AR"] = - 1.78 * 0.045 * 0.68 * AR_w ** (0.68 - 1)


class InducedDragConstant(om.ExplicitComponent):
    """
    Computes the induced drag constant
    """

    def setup(self):
        self.add_input("data:geometry:wing:AR", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CDi:e", val=np.nan, units=None)
        self.add_output("data:aerodynamics:CDi:K", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_w = inputs["data:geometry:wing:AR"]
        e = inputs["data:aerodynamics:CDi:e"]

        K = 1 / (np.pi * e * AR_w)  # induced drag constant (correction term for non-elliptical lift distribution)

        outputs["data:aerodynamics:CDi:K"] = K

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        AR_w = inputs["data:geometry:wing:AR"]
        e = inputs["data:aerodynamics:CDi:e"]

        partials["data:aerodynamics:CDi:K",
                 "data:geometry:wing:AR"] = - 1 / (np.pi * e * AR_w ** 2)
        partials["data:aerodynamics:CDi:K",
                 "data:aerodynamics:CDi:e"] = - 1 / (np.pi * e ** 2 * AR_w)
