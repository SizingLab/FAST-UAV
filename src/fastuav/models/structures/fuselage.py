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
        self.add_input("data:weight:airframe:fuselage:mass:density", val=np.nan, units="kg/m**2")
        self.add_output("data:weight:airframe:fuselage:mass", units="kg", lower=0.0)
        self.add_output("data:weight:airframe:fuselage:mass:nose", units="kg", lower=0.0)
        self.add_output("data:weight:airframe:fuselage:mass:mid", units="kg", lower=0.0)
        self.add_output("data:weight:airframe:fuselage:mass:rear", units="kg", lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        S_fus = inputs["data:geometry:fuselage:surface"]
        S_nose = inputs["data:geometry:fuselage:surface:nose"]
        S_mid = inputs["data:geometry:fuselage:surface:mid"]
        S_rear = inputs["data:geometry:fuselage:surface:rear"]
        rho_fus = inputs["data:weight:airframe:fuselage:mass:density"]

        m_nose = S_nose * rho_fus
        m_mid = S_mid * rho_fus
        m_rear = S_rear * rho_fus
        m_fus = S_fus * rho_fus  # mass of fuselage [kg]

        outputs["data:weight:airframe:fuselage:mass"] = m_fus
        outputs["data:weight:airframe:fuselage:mass:nose"] = m_nose
        outputs["data:weight:airframe:fuselage:mass:mid"] = m_mid
        outputs["data:weight:airframe:fuselage:mass:rear"] = m_rear

    def compute_partials(self, inputs, partials):
        S_fus = inputs["data:geometry:fuselage:surface"]
        S_nose = inputs["data:geometry:fuselage:surface:nose"]
        S_mid = inputs["data:geometry:fuselage:surface:mid"]
        S_rear = inputs["data:geometry:fuselage:surface:rear"]
        rho_fus = inputs["data:weight:airframe:fuselage:mass:density"]

        # m_fus = S_fus * rho_fus
        partials["data:weight:airframe:fuselage:mass", "data:geometry:fuselage:surface"] = rho_fus
        partials["data:weight:airframe:fuselage:mass", "data:weight:airframe:fuselage:mass:density"] = S_fus

        # m_nose = S_nose * rho_fus
        partials["data:weight:airframe:fuselage:mass:nose", "data:geometry:fuselage:surface:nose"] = rho_fus
        partials["data:weight:airframe:fuselage:mass:nose", "data:weight:airframe:fuselage:mass:density"] = S_nose

        # m_mid = S_mid * rho_fus
        partials["data:weight:airframe:fuselage:mass:mid", "data:geometry:fuselage:surface:mid"] = rho_fus
        partials["data:weight:airframe:fuselage:mass:mid", "data:weight:airframe:fuselage:mass:density"] = S_mid

        # m_rear = S_rear * rho_fus
        partials["data:weight:airframe:fuselage:mass:rear", "data:geometry:fuselage:surface:rear"] = rho_fus
        partials["data:weight:airframe:fuselage:mass:rear", "data:weight:airframe:fuselage:mass:density"] = S_rear


