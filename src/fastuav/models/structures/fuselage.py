"""
Fuselage Structures and Weights
"""
import openmdao.api as om
import numpy as np


class FuselageStructures(om.ExplicitComponent):
    """
    Computes Fuselage mass
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:surface", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:surface:nose", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:surface:mid", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:surface:rear", val=np.nan, units="m**2")
        self.add_input("data:weights:airframe:fuselage:mass:density", val=np.nan, units="kg/m**2")
        self.add_output("data:weights:airframe:fuselage:mass", units="kg", lower=0.0)
        self.add_output("data:weights:airframe:fuselage:mass:nose", units="kg", lower=0.0)
        self.add_output("data:weights:airframe:fuselage:mass:mid", units="kg", lower=0.0)
        self.add_output("data:weights:airframe:fuselage:mass:rear", units="kg", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        S_fus = inputs["data:geometry:fuselage:surface"]
        S_nose = inputs["data:geometry:fuselage:surface:nose"]
        S_mid = inputs["data:geometry:fuselage:surface:mid"]
        S_rear = inputs["data:geometry:fuselage:surface:rear"]
        rho_fus = inputs["data:weights:airframe:fuselage:mass:density"]

        m_nose = S_nose * rho_fus
        m_mid = S_mid * rho_fus
        m_rear = S_rear * rho_fus
        m_fus = S_fus * rho_fus  # mass of fuselage [kg]

        outputs["data:weights:airframe:fuselage:mass"] = m_fus
        outputs["data:weights:airframe:fuselage:mass:nose"] = m_nose
        outputs["data:weights:airframe:fuselage:mass:mid"] = m_mid
        outputs["data:weights:airframe:fuselage:mass:rear"] = m_rear


